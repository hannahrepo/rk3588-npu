import cv2
import numpy as np
from rknn.api import RKNN

# =================================================================
# Configuration Constants
# =================================================================
RKNN_MODEL = "yolov5s_relu.rknn"  # The model you just generated
IMG_PATH = "./bus.jpg"  # Ensure this image exists in current folder
# BOARD_IP = "192.168.2.100"  # Your RK3588 board IP
DEVICE_ID = "0123456789ABCDEF"
TARGET_PLATFORM = "rk3588"

if __name__ == "__main__":
    # Initialize RKNN object with verbose logging for better debugging
    rknn = RKNN(verbose=True)

    # Step 1: Load the RKNN model directly
    # Since we already converted the model using test.py,
    # we can load the .rknn file directly to save time.
    print(f"--> Loading RKNN model: {RKNN_MODEL}")
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("Load RKNN model failed!")
        exit(ret)

    # Step 2: Init Runtime on the remote board
    print(f"--> Connecting to board via USB (Device ID: {DEVICE_ID})")
    ret = rknn.init_runtime(
        target=TARGET_PLATFORM, device_id=DEVICE_ID, perf_debug=True, eval_mem=True
    )

    if ret != 0:
        print(
            "Init runtime failed! Please check if the USB cable is connected properly."
        )
        exit(ret)

    # Step 3: Image Pre-processing
    # Standard YOLOv5 input: 640x640, RGB format
    orig_img = cv2.imread(IMG_PATH)
    if orig_img is None:
        print(f"Error: Cannot find {IMG_PATH}")
        exit(-1)

    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))

    # Step 4: Run Inference on NPU
    # The image data is sent over the network, processed by NPU, and results sent back.
    print("--> Running inference on NPU...")
    outputs = rknn.inference(inputs=[img])
    print("--> Inference completed successfully!")

    # Step 5: Performance Evaluation (Critical for your 1st Video)
    # This will print the execution time for each layer on the RK3588 NPU.
    print("----------------------------------------------------")
    print("PERFORMANCE EVALUATION")
    print("----------------------------------------------------")
    rknn.eval_perf(is_print=True)

    # Step 6: Memory and Version Profiling (Correct for v2.3.0 Toolkit2)
    print("----------------------------------------------------")
    print("SDK VERSION & MEMORY PROFILING")
    print("----------------------------------------------------")

    # 1. 获取 SDK 版本信息
    sdk_version = rknn.get_sdk_version()
    print(f"SDK Version: {sdk_version}")

    # 2. 评估模型内存占用 (Weight / Internal Memory)
    # 这会打印出模型在 NPU 中占用的系统内存
    rknn.eval_memory()
