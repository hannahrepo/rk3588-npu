import os
import multiprocessing as mp
import time
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# =================================================================
# Configuration
# =================================================================
RKNN_MODEL = "./yolov5s_relu.rknn"
IMG_PATH = "./bus.jpg"
TEST_SECONDS = 20
# Bind to specific NPU cores: [Core0, Core1, Core2]
CORES = [1, 2, 4]


def worker(core_mask, model_path, img_data, counter, stop_event):
    """
    Each process runs this worker function independently.
    """
    # Initialize RKNNLite inside the process to avoid resource sharing
    rknn_lite = RKNNLite()

    # Load model
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print(f"Core {core_mask}: Load model failed")
        return

    # Init runtime with core mask
    ret = rknn_lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        print(f"Core {core_mask}: Init runtime failed")
        return

    print(f"Worker process for Core Mask {core_mask} is ready.")

    # Inference loop
    local_count = 0
    while not stop_event.is_set():
        rknn_lite.inference(inputs=[img_data], data_format=["nhwc"])
        local_count += 1
        if local_count >= 100:  # 减少加锁频率
            with counter.get_lock():
                counter.value += local_count
            local_count = 0

    rknn_lite.release()


if __name__ == "__main__":
    # Ensure images exist
    if not os.path.exists(RKNN_MODEL) or not os.path.exists(IMG_PATH):
        print("Error: Model or Image not found!")
        exit(-1)

    # 1. Pre-process image once in main process
    img_src = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img_input = np.expand_dims(img, 0)

    # 2. Setup shared variables
    # 'i' for integer, initial value 0
    total_frames = mp.Value("i", 0)
    stop_event = mp.Event()

    # 3. Spawn 3 processes (one for each core)
    # Note: If 3 processes don't hit 100% load, you can spawn 6 (2 per core)
    processes = []
    print(f"--> Spawning {len(CORES)} independent processes...")

    for mask in CORES:
        # For even more pressure, you can loop this twice to have 2 proc/core
        for _ in range(2):
            p = mp.Process(
                target=worker,
                args=(mask, RKNN_MODEL, img_input, total_frames, stop_event),
            )
            p.start()
            processes.append(p)

    # 4. Start timing
    print(f"--> Stress test started! Running for {TEST_SECONDS} seconds...")
    start_time = time.time()

    try:
        time.sleep(TEST_SECONDS)
    except KeyboardInterrupt:
        print("Interrupted by user.")

    # 5. Cleanup
    stop_event.set()
    for p in processes:
        p.join()

    end_time = time.time()
    actual_duration = end_time - start_time
    total_count = total_frames.value
    final_fps = total_count / actual_duration

    print("\n" + "=" * 50)
    print(f"MULTI-PROCESS STRESS TEST RESULT (RK3588)")
    print("=" * 50)
    print(f"Active NPU Cores:       Core 0, 1, 2")
    print(f"Total Frames Processed: {total_count}")
    print(f"Actual Duration:        {actual_duration:.2f} s")
    print(f"System Throughput:      {final_fps:.2f} FPS")
    print(f"Performance per Core:   {final_fps/3:.2f} FPS")
    print("=" * 50)
