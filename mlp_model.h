#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include "mlp_weights.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <stdint.h>

// Declares fixed-point helpers and accelerator-facing APIs for the MLP pipeline.
typedef ap_axiu<32,0,0,0> AXIS_wLAST;

// Multiplies two Q8.8 values and clamps the result to int16 range.
inline MLP_DTYPE mlp_fixed_multiply(MLP_DTYPE a, MLP_DTYPE b) {
    int32_t result = ((int32_t)a * (int32_t)b) >> MLP_SCALE_SHIFT;
    if (result > 32767) result = 32767;
    if (result < -32768) result = -32768;
    return (MLP_DTYPE)result;
}

// Adds two Q8.8 values with saturation.
inline MLP_DTYPE mlp_fixed_add(MLP_DTYPE a, MLP_DTYPE b) {
    int32_t result = (int32_t)a + (int32_t)b;
    if (result > 32767) result = 32767;
    if (result < -32768) result = -32768;
    return (MLP_DTYPE)result;
}

// Applies ReLU in fixed-point domain.
inline MLP_DTYPE mlp_relu(MLP_DTYPE x) {
    return (x > 0) ? x : 0;
}

// Converts float input to Q8.8 with representable-range clamping.
inline MLP_DTYPE mlp_float_to_fixed(float x) {
    if (x > 127.996f) x = 127.996f;
    if (x < -128.0f) x = -128.0f;
    return (MLP_DTYPE)(x * MLP_SCALE_FACTOR);
}

// Converts Q8.8 back to float for software-side interoperability.
inline float mlp_fixed_to_float(MLP_DTYPE x) {
    return ((float)x) / MLP_SCALE_FACTOR;
}

// Top-level AXI-stream entry used for HLS synthesis.
void mlp_gesture_detection(
    hls::stream<AXIS_wLAST> &input_stream,
    hls::stream<AXIS_wLAST> &output_stream);

// Fixed-point helper kernels used in the forward pass.
void mlp_relu_activation(MLP_DTYPE* data, int size);
void mlp_softmax_activation(MLP_DTYPE* data, int size);
int mlp_argmax(MLP_DTYPE* data, int size);
void mlp_dense_layer(
    const MLP_DTYPE* input, 
    const MLP_DTYPE* weights,
    const MLP_DTYPE* bias,
    MLP_DTYPE* output,
    int input_size,
    int output_size);

// Generic matrix/vector helpers kept for reuse and benchmarking.
void mlp_matrix_vector_mult(const MLP_DTYPE* W, const MLP_DTYPE* x, const MLP_DTYPE* b,
                           MLP_DTYPE* y, int rows, int cols);
void mlp_apply_relu(MLP_DTYPE* vector, int size);

#endif

