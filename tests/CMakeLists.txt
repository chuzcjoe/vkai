file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")
add_executable(vkai_tests ${test_SRC})

target_link_libraries(vkai_tests PRIVATE vkai_tasks gtest_main)

target_compile_definitions(vkai_tests
    PRIVATE VKAI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_test(NAME vkai_tests COMMAND vkai_tests)
