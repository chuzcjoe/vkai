#!/usr/bin/env bash

set -euo pipefail

git config core.hooksPath .githooks

mkdir -p build
cd build 

project_root="$(cd .. && pwd)"
cmake_options=(-DCMAKE_BUILD_TYPE=Debug
               -DPIPELINE_CACHE_DIR="$project_root"
               -DENABLE_EXTERNAL=1
               -DENABLE_VULKAN=1
               -DENABLE_OPENCL=0
               -DENABLE_MAT=0
               -DENABLE_TIMER=0
               -DENABLE_TRACE=0
               -DENABLE_OPENGL=0
               -DENABLE_METAL=0
               -DENABLE_EGL=0
               -DENABLE_IO=0
               -DENABLE_THREADPOOL=0
               -DENABLE_TESTS=0
               -DENABLE_EXAMPLES=0)

cmake "${cmake_options[@]}" ..
make -j10

# test
./tests/vkai_tests