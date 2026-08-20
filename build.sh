#!/usr/bin/env bash

set -euo pipefail

git config core.hooksPath .githooks

mkdir -p build
cd build 

cmake ..
make

# test
./tests/vkai_tests