cmake_minimum_required(VERSION 3.16)
project(VKAI CXX)

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

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE vkai_flatbuffers FlatBuffers::FlatBuffers)
