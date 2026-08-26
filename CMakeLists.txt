cmake_minimum_required(VERSION 3.16)
project(VKAI CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(external)

# Keep project warnings away from third-party targets in external/.
add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wunused-variable
  -Wunused-function
  -Wshadow
  -Werror
  -O3
)

flatbuffers_generate_headers(
  TARGET vkai_flatbuffers
  SCHEMAS schemas/model.fbs
)

flatbuffers_generate_binary_files(
  TARGET vkai_weights
  SCHEMA schemas/model.fbs
  JSON_FILES assets/weights.json
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

# add_definitions(-DPIPELINE_CACHE_DIR="${PIPELINE_CACHE_DIR}")
add_definitions(-DPIPELINE_CACHE_DIR="")

add_subdirectory(vkai)
add_subdirectory(tests)
