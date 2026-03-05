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