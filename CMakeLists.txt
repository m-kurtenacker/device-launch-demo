cmake_minimum_required(VERSION 3.25)

project(Test LANGUAGES CXX CUDA)

find_package(SDL2 REQUIRED)
find_package (AnyDSL_runtime REQUIRED)

include_directories(${AnyDSL_runtime_INCLUDE_DIRS})

anydsl_runtime_wrap(TEST_MANDELBROT_OBJ
    FRONTEND "artic"
    FILES test_mandelbrot.art
)

anydsl_runtime_wrap(TEST_SIERPINSKI_OBJ
    FRONTEND "artic"
    FILES test_sierpinski.art
)

add_executable(mandelbrot_view main.cpp ${TEST_MANDELBROT_OBJ})
target_include_directories(mandelbrot_view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(mandelbrot_view ${SDL2_LIBRARIES})
target_link_libraries(mandelbrot_view ${AnyDSL_runtime_LIBRARIES})

set_property(TARGET mandelbrot_view
    APPEND PROPERTY ADDITIONAL_CLEAN_FILES test_mandelbrot.nvvm test_mandelbrot.cu test_mandelbrot.amdgpu test_mandelbrot.amdgpu.obj test_mandelbrot.amdgpu.gcn test_mandelbrot.nvvm.bc test_mandelbrot.l0.spv test_mandelbrot.cl.spv test_mandelbrot.vk.spv cache)

add_executable(sierpinski_view main.cpp ${TEST_SIERPINSKI_OBJ})
target_include_directories(sierpinski_view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(sierpinski_view ${SDL2_LIBRARIES})
target_link_libraries(sierpinski_view ${AnyDSL_runtime_LIBRARIES})

set_property(TARGET sierpinski_view
    APPEND PROPERTY ADDITIONAL_CLEAN_FILES test_sierpinski.nvvm test_sierpinski.cu test_sierpinski.amdgpu test_sierpinski.amdgpu.obj test_sierpinski.amdgpu.gcn test_sierpinski.nvvm.bc test_sierpinski.l0.spv test_sierpinski.cl.spv test_sierpinski.vk.spv cache)

add_executable(mandelbrot_cuda_view main.cpp test_mandelbrot_cuda.cu)
target_include_directories(mandelbrot_cuda_view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(mandelbrot_cuda_view ${SDL2_LIBRARIES})
set_target_properties(mandelbrot_cuda_view PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(sierpinski_cuda_view main.cpp test_sierpinski_cuda.cu)
target_include_directories(sierpinski_cuda_view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(sierpinski_cuda_view ${SDL2_LIBRARIES})
set_target_properties(sierpinski_cuda_view PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
