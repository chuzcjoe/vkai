#!/usr/bin/env bash

set -euo pipefail

mkdir -p build
cd build 

cmake ..
make

# test
./tests/vkai_tests