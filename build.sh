#!/usr/bin/env bash

set -euo pipefail

git config core.hooksPath .githooks

mkdir -p build
cd build 

project_root="$(cd .. && pwd)"
cmake_options=(-DCMAKE_BUILD_TYPE=Debug
               -DPIPELINE_CACHE_DIR="$project_root")

cmake "${cmake_options[@]}" ..
make -j10

# test
./tests/vkai_tests