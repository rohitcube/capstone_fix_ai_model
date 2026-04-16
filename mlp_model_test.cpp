#include "mlp_model.h"
#include "mlp_test_data.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "hls_stream.h"

/**
 * @brief Bridge helper that lets this host-style test call the stream-based accelerator API.
 *
 * The hardware entrypoint consumes and produces AXI stream words. This wrapper keeps the
 * test harness simple by accepting plain float arrays and handling the packing/unpacking.
 *
 * @param input_data  Floating-point feature vector of length MLP_INPUT_SIZE.
 * @param output_data Output logits buffer of length MLP_OUTPUT_SIZE.
 */
void mlp_gesture_detection_wrapper(float input_data[MLP_INPUT_SIZE], float output_data[MLP_OUTPUT_SIZE]) {
    hls::stream<AXIS_wLAST> input_stream;
    hls::stream<AXIS_wLAST> output_stream;
    
    // Push all input features into the AXI input stream.
    for (int i = 0; i < MLP_INPUT_SIZE; i++) {
        AXIS_wLAST temp;
        // Bit-cast float payload into a 32-bit transfer word.
        union { float f; uint32_t i; } converter;
        converter.f = input_data[i];
        temp.data = converter.i;
        temp.last = (i == MLP_INPUT_SIZE - 1) ? 1 : 0;
        input_stream.write(temp);
    }
    
    // Run inference on the DUT.
    mlp_gesture_detection(input_stream, output_stream);
    
    // Read logits back from the AXI output stream.
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
        AXIS_wLAST temp = output_stream.read();
        // Reconstruct float value from the 32-bit transfer word.
        union { float f; uint32_t i; } converter;
        converter.i = temp.data;
        output_data[i] = converter.f;
    }
}

/**
 * @brief Pretty-prints a float array with fixed precision.
 *
 * @param name  Label prefix shown before the array contents.
 * @param arr   Pointer to the float array to display.
 * @param size  Number of elements to print.
 */
void print_array(const char* name, const float* arr, int size) {
    std::cout << name << ": [";
    for (int i = 0; i < size; i++) {
        std::cout << std::fixed << std::setprecision(8) << arr[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Returns the index of the maximum value in a float array.
 *
 * @param data Pointer to the array.
 * @param size Number of elements in the array.
 * @return Index of the largest element.
 */
int argmax_float(const float* data, int size) {
    float max_val = data[0];
    int max_idx = 0;
    
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/**
 * @brief Entry point for software-side regression of the MLP hardware model.
 *
 * Runs a fixed number of curated test vectors, prints per-sample predictions, and reports
 * aggregate accuracy at the end.
 *
 * @return Exit code (0 on completion).
 */
int main() {
    std::cout << "=== MLP Gesture Detection Test - 2-IMU Test Model ===" << std::endl;
    std::cout << "Model Architecture: " << MLP_INPUT_SIZE << " -> " 
              << MLP_LAYER1_SIZE << " -> " << MLP_LAYER2_SIZE 
              << " -> " << MLP_OUTPUT_SIZE << std::endl << std::endl;
    
    const char* gesture_names[MLP_OUTPUT_SIZE] = {
        "no_gesture", "move_forward", "turn_left", "turn_right",
        "jump", "attack", "turn_180",
        "dummy"
    };

    float output[MLP_OUTPUT_SIZE];

    int correct_predictions = 0;
    int total_predictions = TEST_SAMPLES;
    
    for (int sample = 0; sample < total_predictions; sample++) {
        std::cout << "--- Test Sample " << (sample + 1) << " ---" << std::endl;
        
        // Copy current sample into a mutable local buffer for the wrapper call.
        float input_data[MLP_INPUT_SIZE];
        for (int i = 0; i < MLP_INPUT_SIZE; i++) {
            input_data[i] = test_input_data_float[sample][i];
        }
        
        // Execute accelerator inference.
        mlp_gesture_detection_wrapper(input_data, output);
        
        // Print raw inputs/logits and expected logits for quick visual debugging.
        print_array("Input (first 10)", input_data, 10);
        print_array("Output logits", output, MLP_OUTPUT_SIZE);
        print_array("Expected", (float*)test_expected_output[sample], MLP_OUTPUT_SIZE);
        
        // Compare predicted class IDs from argmax over logits.
        int predicted_class = argmax_float(output, MLP_OUTPUT_SIZE);
        int expected_class = argmax_float(test_expected_output[sample], MLP_OUTPUT_SIZE);
        
        std::cout << "Predicted: " << gesture_names[predicted_class] 
                  << " (class " << predicted_class << ")" << std::endl;
        std::cout << "Expected:  " << gesture_names[expected_class] 
                  << " (class " << expected_class << ")" << std::endl;
        
        if (predicted_class == expected_class) {
            std::cout << "CORRECT" << std::endl;
            correct_predictions++;
        } else {
            std::cout << "INCORRECT" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Accuracy: " << correct_predictions << "/" << total_predictions 
              << " (" << (100.0 * correct_predictions / total_predictions) << "%)" << std::endl;
    
    return 0;
}

