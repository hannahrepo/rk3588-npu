import threading
import time
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# =================================================================
# Board-Side Single Core Stress Test Configuration
# =================================================================
RKNN_MODEL = "./yolov5s_relu.rknn"
IMG_PATH = "./bus.jpg"
NUM_THREADS = 16
TEST_SECONDS = 20
CORES = [1]  # Only Core 0 (Mask = 1)

total_frames = 0
counter_lock = threading.Lock()


def inference_thread(rknn_instance, lock, thread_id, img_data, stop_event):
    global total_frames
    print(f"Thread-{thread_id}: Active on Core 0")
    while not stop_event.is_set():
        # All threads will share this single core instance
        with lock:
            rknn_instance.inference(inputs=[img_data], data_format=["nhwc"])
        with counter_lock:
            total_frames += 1


if __name__ == "__main__":
    instances = []
    locks = []

    # 1. Image loading
    img_src = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img_input = np.expand_dims(img, 0)

    # 2. Initialize Instance for the single NPU core 0
    print("--> Initializing 1 RKNNLite instance for Core 0...")
    for i, mask in enumerate(CORES):
        rknn = RKNNLite()
        if rknn.load_rknn(RKNN_MODEL) != 0:
            print(f"Load model failed for Core 0")
            exit(-1)
        # Bind specifically to Core 0
        if rknn.init_runtime(core_mask=mask) != 0:
            print(f"Init runtime failed for Core 0")
            exit(-1)
        instances.append(rknn)
        locks.append(threading.Lock())

    # 3. Spawn 16 threads to saturate the single core
    threads = []
    stop_event = threading.Event()
    start_run_time = time.time()

    for i in range(NUM_THREADS):
        # All threads use the same instance (instances[0])
        instance_idx = 0
        t = threading.Thread(
            target=inference_thread,
            args=(
                instances[instance_idx],
                locks[instance_idx],
                i,
                img_input,
                stop_event,
            ),
        )
        threads.append(t)
        t.start()

    print(f"--> Performance test running on SINGLE CORE for {TEST_SECONDS}s...")
    try:
        time.sleep(TEST_SECONDS)
    except KeyboardInterrupt:
        pass

    stop_event.set()
    for t in threads:
        t.join()

    actual_time = time.time() - start_run_time
    final_fps = total_frames / actual_time

    print("\n" + "=" * 50)
    print(f"SINGLE CORE PERFORMANCE RESULT")
    print("=" * 50)
    print(f"Total Inference Cycles: {total_frames}")
    print(f"Total Duration:         {actual_time:.2f} s")
    print(f"Throughput (FPS):       {final_fps:.2f} FPS")
    print("=" * 50)

    for rknn in instances:
        rknn.release()
