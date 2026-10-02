cmake_minimum_required(VERSION 3.25)

project (Test)

find_package(SDL2 REQUIRED)
find_package (AnyDSL_runtime REQUIRED)

include_directories(${AnyDSL_runtime_INCLUDE_DIRS})

anydsl_runtime_wrap(TEST_OBJ
    FRONTEND "artic"
    FILES test.art
)

add_executable(mandelbrot_view main.cpp ${TEST_OBJ})
target_include_directories(mandelbrot_view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(mandelbrot_view ${SDL2_LIBRARIES})
target_link_libraries(mandelbrot_view ${AnyDSL_runtime_LIBRARIES})

set_property(TARGET mandelbrot_view
    APPEND PROPERTY ADDITIONAL_CLEAN_FILES test.nvvm test.cu test.amdgpu test.amdgpu.obj test.amdgpu.gcn test.nvvm.bc test.l0.spv test.cl.spv test.vk.spv cache)
