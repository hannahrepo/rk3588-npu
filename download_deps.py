import subprocess
import os
import sys

# =================================================================
# Configuration for Offline Dependency Downloader
# =================================================================

# Path to the local RKNN-Lite2 wheel file
WHL_PATH = "./rknn-toolkit2-2.3.0/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

# Directory where dependencies will be stored
DEPS_DIR = "./deps"

# Target platform and Python version
PLATFORM = "manylinux2014_aarch64"
PYTHON_VER = "310"

# Explicitly add packages that might be missed by automatic resolution
EXTRA_PACKAGES = ["pillow>=9.3.0", "opencv-python-headless"]


def download_dependencies():
    if not os.path.exists(WHL_PATH):
        print(f"Error: Target wheel file not found at {WHL_PATH}")
        return

    if not os.path.exists(DEPS_DIR):
        os.makedirs(DEPS_DIR)

    # 1. Automatic dependency resolution from the local wheel
    print(f"--> [Phase 1] Downloading core dependencies from wheel...")
    base_cmd = [
        "pip3",
        "download",
        "--platform",
        PLATFORM,
        "--only-binary=:all:",
        "--python-version",
        PYTHON_VER,
        "-d",
        DEPS_DIR,
    ]

    try:
        subprocess.run(base_cmd + [WHL_PATH], check=True)

        # 2. Force download extra packages like Pillow
        print(f"--> [Phase 2] Downloading extra packages: {EXTRA_PACKAGES}...")
        for pkg in EXTRA_PACKAGES:
            subprocess.run(base_cmd + [pkg], check=True)

        print("\n" + "=" * 50)
        print("SUCCESS: All mandatory and extra dependencies are in ./deps")
        print("=" * 50)

    except subprocess.CalledProcessError as e:
        print(f"\nError: pip download failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    download_dependencies()
