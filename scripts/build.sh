#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 [-r {unittests|tasks}]" >&2
}

test_mode="unittests"
if [[ $# -eq 0 ]]; then
  :
elif [[ $# -eq 2 && "$1" == "-r" ]]; then
  test_mode="$2"
else
  usage
  exit 2
fi

case "$test_mode" in
  unittests)
    test_target="vkai_unittests"
    test_command=(./tests/vkai_unittests)
    ;;
  tasks)
    test_target="vkai_tasks_tests"
    test_command=(./tasks/vkai_tasks_tests)
    ;;
  *)
    usage
    exit 2
    ;;
esac

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
cmake --build . --target "$test_target" -j10

"${test_command[@]}"
