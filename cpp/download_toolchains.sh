#!/usr/bin/env bash
# Download the Arm GNU 9.2 aarch64 toolchain and the AX650 BSP SDK (msp/out).
# Everything lands in cpp/toolchains/ (git-ignored).
set -euo pipefail

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAINS_DIR="${CPP_DIR}/toolchains"

GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
GCC_DIR="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

if [[ -x "${TOOLCHAINS_DIR}/${GCC_DIR}/bin/aarch64-none-linux-gnu-g++" ]]; then
  echo "Found ${TOOLCHAINS_DIR}/${GCC_DIR}"
else
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  echo "Downloading ${GCC_URL}"
  wget -q --show-progress "${GCC_URL}" -O gcc.tar.xz
  tar -xf gcc.tar.xz
  rm -f gcc.tar.xz
  echo "Done: ${TOOLCHAINS_DIR}/${GCC_DIR}"
fi

if [[ -f "${TOOLCHAINS_DIR}/ax650n_bsp_sdk/msp/out/include/ax_engine_api.h" ]]; then
  echo "Found ${TOOLCHAINS_DIR}/ax650n_bsp_sdk"
else
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  echo "Cloning https://github.com/AXERA-TECH/ax650n_bsp_sdk.git"
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
  echo "Done: ${TOOLCHAINS_DIR}/ax650n_bsp_sdk"
fi
