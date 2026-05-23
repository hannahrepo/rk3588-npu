import os
import sys
from rkllm.api import RKLLM

# ================= Configuration =================
# Path to the exported .rkllm file
MODEL_PATH = "./Qwen2.5-3B-Instruct"
RKLLM_OUTPUT_PATH = "./qwen2.5_3b_instruct_w8a8.rkllm"
# Target hardware platform
TARGET_PLATFORM = "rk3588"
# =================================================


def main():
    rkllm = RKLLM()

    # Step 1: Load the original HuggingFace model
    print(f"--> Loading HuggingFace model from: {MODEL_PATH}")
    # Force loading on CPU to avoid CUDA Out of Memory error on old GPUs like GT730
    ret = rkllm.load_huggingface(model=MODEL_PATH, device="cpu")
    if ret != 0:
        print("Error: Failed to load the model.")
        sys.exit(ret)

    # Step 2: Build the model and perform INT8 quantization (W8A8)
    print("--> Building RKLLM model with w8a8 (INT8) quantization...")
    ret = rkllm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8",
        target_platform=TARGET_PLATFORM,
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
