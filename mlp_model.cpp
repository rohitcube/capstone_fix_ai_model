#include "mlp_model.h"
#include "mlp_weights.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "hls_stream.h"

// quick relu pass; keeping this separate made timing experiments easier.
void mlp_relu_activation(MLP_DTYPE* data, int size) {
    #pragma HLS INLINE off
    
    RELU_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

// rough softmax approximation for now (good enough for ranking classes).
// TODO: replace with a tighter approximation if probability calibration matters.
void mlp_softmax_activation(MLP_DTYPE* data, int size) {
    #pragma HLS INLINE off

    MLP_DTYPE max_val = data[0];
    
    SOFTMAX_MAX_LOOP:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    int32_t sum_exp = 0;
    
    SOFTMAX_EXP_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        MLP_DTYPE shifted = data[i] - max_val;

        if (shifted < -MLP_SCALE_FACTOR) {
            data[i] = 1;
        } else {
            data[i] = MLP_SCALE_FACTOR + shifted;
            if (data[i] < 1) data[i] = 1;
        }
        sum_exp += data[i];
    }

    SOFTMAX_NORM_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        data[i] = (data[i] * MLP_SCALE_FACTOR) / (sum_exp >> 8);
        if (data[i] > MLP_SCALE_FACTOR) data[i] = MLP_SCALE_FACTOR;
    }
}

// simple argmax utility used by the software-side checks.
int mlp_argmax(MLP_DTYPE* data, int size) {
    #pragma HLS INLINE off
    
    MLP_DTYPE max_val = data[0];
    int max_idx = 0;
    
    ARGMAX_LOOP:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// core dense layer kernel; weights are flattened row-major from export script.
// note: unroll factor is still a tuning knob depending on resource pressure.
void mlp_dense_layer(
    const MLP_DTYPE* input, 
    const MLP_DTYPE* weights,
    const MLP_DTYPE* bias,
    MLP_DTYPE* output,
    int input_size,
    int output_size) {
    #pragma HLS INLINE off
    
    DENSE_OUTPUT_LOOP:
    for (int i = 0; i < output_size; i++) {
        #pragma HLS PIPELINE II=1
        
        int32_t accumulator = 0;
        
        DENSE_INPUT_LOOP:
        for (int j = 0; j < input_size; j++) {
            #pragma HLS UNROLL factor=4
            accumulator += (int32_t)input[j] * (int32_t)weights[j * output_size + i];
        }

        accumulator >>= MLP_SCALE_SHIFT;
        accumulator += (int32_t)bias[i];

        if (accumulator > 32767) accumulator = 32767;
        if (accumulator < -32768) accumulator = -32768;

        output[i] = (MLP_DTYPE)accumulator;
    }
}

// top-level stream interface for HLS IP.
// flow: read features -> run 3 layers -> write logits.
void mlp_gesture_detection(
    hls::stream<AXIS_wLAST> &input_stream,
    hls::stream<AXIS_wLAST> &output_stream) {

#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // scratch buffers for hidden layers.
    MLP_DTYPE layer1_output[MLP_LAYER1_SIZE];
    MLP_DTYPE layer2_output[MLP_LAYER2_SIZE];
    #pragma HLS ARRAY_PARTITION variable=layer1_output complete
    #pragma HLS ARRAY_PARTITION variable=layer2_output complete

    // fixed-point staging buffers (kept local for predictable synthesis).
    MLP_DTYPE input_buffer[MLP_INPUT_SIZE];
    MLP_DTYPE output_buffer[MLP_OUTPUT_SIZE];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete
    #pragma HLS ARRAY_PARTITION variable=output_buffer complete

    // ingress: unpack float bits from DMA stream, then quantize to Q8.8.
    STREAM_INPUT_LOOP:
    for (int i = 0; i < MLP_INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        AXIS_wLAST temp = input_stream.read();
        union { float f; uint32_t i; } converter;
        converter.i = temp.data;
        input_buffer[i] = mlp_float_to_fixed(converter.f);
    }

    // layer 1 (84 -> 64) + relu.
    mlp_dense_layer(input_buffer, 
                   (const MLP_DTYPE*)mlp_dense_64_weights, 
                   mlp_dense_64_bias, 
                   layer1_output, 
                   MLP_INPUT_SIZE, 
                   MLP_LAYER1_SIZE);
    mlp_relu_activation(layer1_output, MLP_LAYER1_SIZE);

    // layer 2 (64 -> 32) + relu.
    mlp_dense_layer(layer1_output, 
                   (const MLP_DTYPE*)mlp_dense_32_weights, 
                   mlp_dense_32_bias, 
                   layer2_output, 
                   MLP_LAYER1_SIZE, 
                   MLP_LAYER2_SIZE);
    mlp_relu_activation(layer2_output, MLP_LAYER2_SIZE);

    // output layer (32 -> 8), keeping raw logits for host-side post-processing.
    mlp_dense_layer(layer2_output, 
                   (const MLP_DTYPE*)mlp_logits_weights, 
                   mlp_logits_bias, 
                   output_buffer, 
                   MLP_LAYER2_SIZE, 
                   MLP_OUTPUT_SIZE);

    // egress: convert back to float bits so host code doesn't need changes.
    // TODO: maybe emit fixed-point directly and dequantize on host if bandwidth gets tight.
    STREAM_OUTPUT_LOOP:
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        AXIS_wLAST temp;
        union { float f; uint32_t i; } converter;
        converter.f = mlp_fixed_to_float(output_buffer[i]);
        temp.data = converter.i;
		temp.keep = -1;
		temp.strb = -1;
		temp.last = (i == MLP_OUTPUT_SIZE - 1) ? 1 : 0;
        output_stream.write(temp);
    }
}

// older generic mat-vec path; not on main hot path right now.
void mlp_matrix_vector_mult(const MLP_DTYPE* W, const MLP_DTYPE* x, const MLP_DTYPE* b,
                           MLP_DTYPE* y, int rows, int cols) {
    #pragma HLS INLINE off
    
    MATRIX_MULT_ROWS:
    for (int i = 0; i < rows; i++) {
        #pragma HLS PIPELINE II=1
        int32_t accumulator = 0;

        MATRIX_MULT_COLS:
        for (int j = 0; j < cols; j++) {
            #pragma HLS UNROLL factor=4
            accumulator += (int32_t)W[i * cols + j] * (int32_t)x[j];
        }

        accumulator >>= MLP_SCALE_SHIFT;
        accumulator += (int32_t)b[i];

        if (accumulator > 32767) accumulator = 32767;
        if (accumulator < -32768) accumulator = -32768;

        y[i] = (MLP_DTYPE)accumulator;
    }
}

// fallback relu helper for shared utility paths.
void mlp_apply_relu(MLP_DTYPE* vector, int size) {
    #pragma HLS INLINE off
    
    APPLY_RELU_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (vector[i] < 0) {
            vector[i] = 0;
        }
    }
}

