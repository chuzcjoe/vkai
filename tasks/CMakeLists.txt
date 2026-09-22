file(GLOB vkai_tasks_SRC CONFIGURE_DEPENDS "src/*.cpp")

add_library(vkai_tasks ${vkai_tasks_SRC})

target_include_directories(vkai_tasks
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(vkai_tasks
    PUBLIC
        vkai
)

file(GLOB vkai_tasks_test_SRC CONFIGURE_DEPENDS "tests/*.cpp")
add_executable(vkai_tasks_tests ${vkai_tasks_test_SRC})

target_link_libraries(vkai_tasks_tests PRIVATE vkai_tasks gtest_main)

target_compile_definitions(vkai_tasks_tests
    PRIVATE VKAI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
