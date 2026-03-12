# RKNN Related Scripts Overview

## 1️⃣ RKNN Development Environment One-Click Deployment Script
**Filename:**  
`setup_rknn.sh`

**Description:**  
Used to automatically deploy the RKNN development environment on the PC side with a single command.

---

## 2️⃣ PC-Side Remote Debugging Script
**Filename:**  
`pc_remote_test_v1.py`

**Description:**  
Used for remotely debugging Python programs running on the target board from the PC.

---

## 3️⃣ RKNN Toolkit Lite2 Dependency Download Script
**Filename:**  
`download_deps.py`

**Related WHL File:**  
`rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

**Description:**  
Used to download and install the required dependencies for RKNN Toolkit Lite2.

---

## 4️⃣ 16-Thread Stress Test Script (Board Side)
**Filename:**  
`board_stress_test_16t.py`

**Description:**  
Used to perform a 16-thread concurrent stress test on the development board.

---

## 5️⃣ 6-Process Stress Test Script (Board Side)
**Filename:**  
`board_stress_test_6p.py`

**Description:**  
Used to perform a 6-process concurrent stress test on the development board.

---

## 6️⃣ C++ High-Performance Zero-Copy Benchmark
**Filename:**
`main.cc`

**Description:**
A native C++ high-performance benchmark tool that implements Zero-Copy technology and multi-core affinity. It maximizes NPU throughput by using high-priority execution (RKNN_FLAG_PRIOR_HIGH) and bypassing CPU memory copy overhead, achieving the ultimate 181+ FPS on RK3588.

---

## 7️⃣ RK3588 Hardware Frequency Locking Script
**Filename:**
`fix_freq_rk3588.sh`

**Description:**
A shell script designed to lock the CPU, GPU, NPU, and DDR frequencies to their maximum values. It also disables CPU idle states to ensure stable and consistent NPU performance during intensive AI workloads.

