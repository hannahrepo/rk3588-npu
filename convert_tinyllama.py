import os
import sys
from rkllm.api import RKLLM

# Configuration
# Path to the downloaded HuggingFace model directory
MODEL_PATH = '../TinyLlama-1.1B-Chat-v1.0' 
# Output .rkllm file path
RKLLM_OUTPUT_PATH = './tinyllama_1.1b_chat_w8a8.rkllm'
# Target platform defined in the manual
TARGET_PLATFORM = 'rk3588'

def main():
    rkllm = RKLLM()

    # Step 1: Load the HuggingFace model
    print(f"--> Loading HuggingFace model from: {MODEL_PATH}")
    ret = rkllm.load_huggingface(model=MODEL_PATH, device='cpu')
    if ret != 0:
        print("Error: Failed to load the model.")
        sys.exit(ret)

    # Step 2: Build the model
    # Based on the manual: RK3588 supports 'w8a8', 'w8a8_g128', etc.
    # 'w8a8' means 8-bit quantization for both weights and activations.
    print("--> Building RKLLM model with w8a8 quantization (Official RK3588 Support)...")
    ret = rkllm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype='w8a8', # Corrected based on the technical manual
        target_platform=TARGET_PLATFORM
    )
    if ret != 0:
        print("Error: Build model failed. Please check memory and target settings.")
        sys.exit(ret)

    # Step 3: Export the compiled .rkllm file
    print(f"--> Exporting quantized model to: {RKLLM_OUTPUT_PATH}")
    ret = rkllm.export_rkllm(RKLLM_OUTPUT_PATH)
    if ret != 0:
        print("Error: Failed to export the .rkllm file.")
        sys.exit(ret)

    print(f"--> SUCCESS! Output file: {RKLLM_OUTPUT_PATH}")

if __name__ == "__main__":
    main()