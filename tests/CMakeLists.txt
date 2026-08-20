file(GLOB test_SRC "*.cpp")
add_executable(vkai_tests ${test_SRC})

target_link_libraries(vkai_tests PRIVATE vkai gtest_main)
