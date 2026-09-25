file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")
add_executable(vkai_unittests ${test_SRC})

target_link_libraries(vkai_unittests PRIVATE vkai gtest_main vkai_flatbuffers)

target_include_directories(vkai_unittests
    PRIVATE
        "${CMAKE_SOURCE_DIR}/external/flatbuffers/include"
)

add_dependencies(vkai_unittests GENERATE_vkai_flatbuffers vkai_weights_target)

target_compile_definitions(vkai_unittests
    PRIVATE
        VKAI_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        VKAI_BINARY_DIR="${CMAKE_BINARY_DIR}")
