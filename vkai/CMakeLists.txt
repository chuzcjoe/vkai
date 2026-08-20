file(GLOB vkai_SRC "src/*.cpp")

add_library(vkai ${vkai_SRC})

target_include_directories(vkai
    PUBLIC
    include)

target_link_libraries(vkai
    PRIVATE
    vulkan
)