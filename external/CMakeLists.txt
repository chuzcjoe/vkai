if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/CMakeLists.txt")
  message(FATAL_ERROR "Missing dependencies. Run: git submodule update --init")
endif()

# Build only the FlatBuffers compiler and header-only C++ runtime.
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)
