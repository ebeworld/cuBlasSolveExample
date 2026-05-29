cmake_minimum_required(VERSION 3.18)

project(CUDALinearEquationSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(solve_linear_equation
    solve_linear_equation.cu
)

set_target_properties(solve_linear_equation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(solve_linear_equation
    PRIVATE
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
)