file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")
add_executable(vkai_unittests ${test_SRC})

target_link_libraries(vkai_unittests PRIVATE vkai gtest_main)

target_compile_definitions(vkai_unittests
    PRIVATE VKAI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
