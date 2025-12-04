cmake_minimum_required(VERSION 4.0)
project(SudokuSolverCuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(SudokuSolverCuda main.cu
        sudoku.h
        args_parser.h
        cpu_solver/cpu_main.h
        gpu_solver/gpu_main.cuh
        cpu_solver/cpu_iterative_solver.h
        gpu_solver/gpu_solver.cuh
        portable_functions.h)

set_target_properties(SudokuSolverCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(SudokuSolverCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -lineinfo
            -Xptxas=-v
        >
)

if(MSVC)
    target_compile_options(SudokuSolverCuda PRIVATE -Xcompiler "/W4")
else()
    target_compile_options(SudokuSolverCuda PRIVATE
            -Xcompiler -Wall
            -Xcompiler -Wextra
    )
endif()