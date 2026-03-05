#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <fstream>
#include <string>

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

/* --- Configuration --- */
#define NUM_CORES 3
#define NUM_THREADS 8    // Adjusted to 8 threads as requested
#define TEST_SECONDS 20

/* --- Global State --- */
std::atomic<long> total_frames(0);
bool stop_test = false;

/* --- Hardware Frequency Helper --- */
double get_sys_freq_ghz(const char* path, double divider) {
    std::ifstream file(path);
    std::string line;
    if (file && std::getline(file, line)) {
        try {
            return std::stod(line) / divider;
        } catch (...) {
            return 0.0;
        }
    }
    return 0.0;
}

static unsigned char* load_data(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error: Cannot open %s\n", filename);
        return NULL;
    }
    fseek(fp, 0L, SEEK_END);
    int n = ftell(fp);
    unsigned char* data = (unsigned char*)malloc(n);
    fseek(fp, 0L, SEEK_SET);
    fread(data, 1, n, fp);
    fclose(fp);
    *model_size = n;
    return data;
}

/* --- Thread Worker: Pure Inference Loop --- */
void inference_worker(rknn_context ctx, int thread_id) {
    while (!stop_test) {
        // High-speed run with Zero-Copy mapped memory
        if (rknn_run(ctx, NULL) < 0) {
            break;
        }
        total_frames.fetch_add(1, std::memory_order_relaxed);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <rknn_model> <input_image_path>\n", argv[0]);
        return -1;
    }

    printf("\n==================================================\n");
    printf("     RK3588 NPU HARDCORE BENCHMARK V1.5\n");
    printf("==================================================\n");

    int ret;
    int model_data_size = 0;
    unsigned char* model_data = load_data(argv[1], &model_data_size);
    if (!model_data) return -1;

    rknn_context ctxs[NUM_CORES];
    rknn_core_mask core_masks[NUM_CORES] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    rknn_tensor_mem* input_mems[NUM_CORES];

    printf("--> Initializing NPU contexts with High Priority...\n");
    for (int i = 0; i < NUM_CORES; i++) {
        // Init with HIGH PRIORITY flag
        ret = rknn_init(&ctxs[i], model_data, model_data_size, RKNN_FLAG_PRIOR_HIGH, NULL);
        if (ret < 0) return -1;
        rknn_set_core_mask(ctxs[i], core_masks[i]);

        rknn_input_output_num io_num;
        rknn_query(ctxs[i], RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        
        rknn_tensor_attr input_attrs[1];
        input_attrs[0].index = 0;
        rknn_query(ctxs[i], RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));

        // ZERO-COPY setup
        input_mems[i] = rknn_create_mem(ctxs[i], input_attrs[0].size);
        
        cv::Mat img = cv::imread(argv[2], 1);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(input_attrs[0].dims[2], input_attrs[0].dims[1]));
        memcpy(input_mems[i]->virt_addr, img.data, input_attrs[0].size);

        // Optimization: Set pass_through to 1 for Zero-Copy performance
        input_attrs[0].pass_through = 1; 
        rknn_set_io_mem(ctxs[i], input_mems[i], &input_attrs[0]);
    }

    // Hardware Stats
    double n_freq = get_sys_freq_ghz("/sys/class/devfreq/fdab0000.npu/cur_freq", 1000000000.0);
    double d_freq = get_sys_freq_ghz("/sys/class/devfreq/dmc/cur_freq", 1000000.0);

    printf("\n[SYSTEM CONFIG]\n");
    printf("NPU Cores:       3 Active\n");
    if (n_freq > 0) printf("NPU Frequency:   %.2f GHz\n", n_freq);
    if (d_freq > 0) printf("DDR Frequency:   %.2f MHz\n", d_freq);
    printf("Threads:         %d\n", NUM_THREADS);

    std::vector<std::thread> workers;
    auto start_time = std::chrono::high_resolution_clock::now();

    printf("\n--> Stress Testing...");
    fflush(stdout);

    for (int i = 0; i < NUM_THREADS; i++) {
        workers.emplace_back(inference_worker, ctxs[i % NUM_CORES], i);
    }

    std::this_thread::sleep_for(std::chrono::seconds(TEST_SECONDS));
    stop_test = true;

    for (auto& t : workers) t.join();
    printf(" DONE!\n");

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double final_fps = (double)total_frames.load() / elapsed.count();
    // Formula for Average Core Latency: (1000ms * Threads) / (Total_FPS * Cores)
    double avg_latency = (1000.0 * NUM_THREADS) / (final_fps * NUM_CORES);

    /* --- Final Report Output (Matched to your template) --- */
    printf("\n--------------------------------------------------\n");
    printf("             FINAL PERFORMANCE REPORT\n");
    printf("--------------------------------------------------\n");
    printf("Total Frames:           %ld\n", total_frames.load());
    printf("Actual Time:            %.2f s\n", elapsed.count());
    printf("System Throughput:      \033[1;32m%.2f FPS\033[0m\n", final_fps);
    printf("Average Core Latency:   %.2f ms\n", avg_latency);
    printf("--------------------------------------------------\n\n");

    for (int i = 0; i < NUM_CORES; i++) {
        rknn_destroy_mem(ctxs[i], input_mems[i]);
        rknn_destroy(ctxs[i]);
    }
    free(model_data);
    return 0;
}