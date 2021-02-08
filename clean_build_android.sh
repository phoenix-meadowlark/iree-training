#!/usr/bin/bash
# Assumes that iree is cloned alongside this directory.
set -x

unset IREE_LLVMAOT_LINKER_PATH

# Hack IREE's tools to allow printing larger buffers for comparison.
sed 's/\/\*max_element_count=\*\/1024/\/*max_element_count=*\/10000000/g' -i \
  ../iree/iree/tools/utils/vm_util.cc

# rm -r ../iree-build-host
cmake -G Ninja -B ../iree-build-host/ \
  -DIREE_ENABLE_CCACHE=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIRS=/usr/include/python3.8 \
  -DCMAKE_INSTALL_PREFIX=../iree-build-host/install \
  ../iree
cmake --build ../iree-build-host/ --target install

# rm -r ../iree-build-android
cmake -G Ninja -B ../iree-build-android/ \
  -DIREE_ENABLE_CCACHE=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT=$(realpath ../iree-build-host/install) \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM=android-29 \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_ENABLE_MLIR=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  ../iree
cmake --build ../iree-build-android/

# Unhack IREE's tools.
sed 's/\/\*max_element_count=\*\/10000000/\/*max_element_count=*\/1024/g' -i \
  ../iree/iree/tools/utils/vm_util.cc
