cmake_minimum_required(VERSION 3.25)

project(Test LANGUAGES CXX CUDA)

find_package(SDL2 REQUIRED)
find_package (AnyDSL_runtime)

if (AnyDSL_runtime_FOUND)
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
endif()

add_executable(mandelbrot_cuda_view main.cpp test_mandelbrot_cuda.cu)
target_include_directories(mandelbrot_cuda_view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(mandelbrot_cuda_view ${SDL2_LIBRARIES})
set_target_properties(mandelbrot_cuda_view PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(sierpinski_cuda_view main.cpp test_sierpinski_cuda.cu)
target_include_directories(sierpinski_cuda_view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(sierpinski_cuda_view ${SDL2_LIBRARIES})
set_target_properties(sierpinski_cuda_view PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# SYCL (Intel oneAPI) port. The SYCL translation unit is compiled with icpx
# into a shared library, since the project's CXX compiler is not SYCL-capable.
find_program(SYCL_COMPILER icpx HINTS /opt/intel/oneapi/compiler/latest/bin)
find_program(DPCT_EXECUTABLE dpct)
if (DPCT_EXECUTABLE)
    get_filename_component(DPCT_BIN_DIR ${DPCT_EXECUTABLE} DIRECTORY)
endif()
find_path(DPCT_INCLUDE_DIR dpct/dpct.hpp HINTS ${DPCT_BIN_DIR}/../include)
set(SYCL_TARGETS "" CACHE STRING "Value for -fsycl-targets (e.g. spir64_gen), empty for JIT")

if (SYCL_COMPILER AND DPCT_INCLUDE_DIR)
    get_filename_component(SYCL_COMPILER_DIR ${SYCL_COMPILER} DIRECTORY)
    set(SYCL_FLAGS -fsycl)
    if (SYCL_TARGETS)
        list(APPEND SYCL_FLAGS -fsycl-targets=${SYCL_TARGETS})
    endif()

    # Builds lib<source>.so from a SYCL source with icpx and links it into a
    # viewer executable built from main.cpp.
    function(add_sycl_view target source)
        set(lib ${CMAKE_CURRENT_BINARY_DIR}/lib${source}.so)
        add_custom_command(
            OUTPUT ${lib}
            COMMAND ${SYCL_COMPILER} ${SYCL_FLAGS} -O3 -fPIC -shared
                    -I${DPCT_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${source}.dp.cpp
                    -o ${lib}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.dp.cpp
            COMMENT "Compiling ${source}.dp.cpp with icpx"
            VERBATIM)
        add_custom_target(${source} DEPENDS ${lib})

        add_executable(${target} main.cpp)
        add_dependencies(${target} ${source})
        target_include_directories(${target} PRIVATE ${SDL2_INCLUDE_DIRS})
        target_link_libraries(${target} ${SDL2_LIBRARIES} ${lib})
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${SYCL_COMPILER_DIR}/../lib")
    endfunction()

    add_sycl_view(mandelbrot_sycl_view test_mandelbrot_sycl)
    add_sycl_view(sierpinski_sycl_view test_sierpinski_sycl)
    add_sycl_view(sierpinski_sycl_persistent_view test_sierpinski_sycl_persistent)
endif()
