import os
import sys
from rkllm.api import RKLLM

# ================= Configuration =================
# 指向你的 DeepSeek-R1-Distill-Qwen-1.5B 模型目录
MODEL_PATH = "./DeepSeek-R1-Distill-Qwen-1.5B"

# 导出的专属 .rkllm 文件名 (加上 w8a8 标记以示区分)
RKLLM_OUTPUT_PATH = "./deepseek_r1_qwen_1.5b_w8a8.rkllm"

# 目标硬件平台 (严格指定 rk3588)
TARGET_PLATFORM = "rk3588"
# =================================================


def main():
    # 初始化 RKLLM 工具
    rkllm = RKLLM()

    # Step 1: 加载 HuggingFace 原版模型
    print(f"--> [1/3] Loading HuggingFace model from: {MODEL_PATH}")
    # 依然强制使用 CPU 加载，避免爆显存
    ret = rkllm.load_huggingface(model=MODEL_PATH, device="cpu")
    if ret != 0:
        print("Error: Failed to load the model. Please check the path.")
        sys.exit(ret)

    # Step 2: 构建模型并进行 INT8 量化 (W8A8)
    # 注意：RK3588 硬件原生支持 W8A8，不支持 W4A16
    print("--> [2/3] Building RKLLM model with w8a8 (INT8) quantization...")
    print(
        "    This step integrates Qwen architecture and DeepSeek reasoning optimizations."
    )
    ret = rkllm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8_g128",  # <--- 核心：RK3588 的唯一真理
        target_platform=TARGET_PLATFORM,
        dataset="./dataset.json",
    )
    if ret != 0:
        print(
            "Error: Build model failed. Please check your toolkit version and memory."
        )
        sys.exit(ret)

    # Step 3: 导出编译好的 .rkllm 文件
    print(f"--> [3/3] Exporting quantized model to: {RKLLM_OUTPUT_PATH}")
    ret = rkllm.export_rkllm(RKLLM_OUTPUT_PATH)
    if ret != 0:
        print("Error: Failed to export the .rkllm file.")
        sys.exit(ret)

    print(f"--> SUCCESS! The DeepSeek model is ready: {RKLLM_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
