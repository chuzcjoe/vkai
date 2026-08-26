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
