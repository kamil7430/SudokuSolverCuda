cmake_minimum_required(VERSION 4.0)
project(SudokuSolverCuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(SudokuSolverCuda main.cu
        sudoku.h
        args_parser.h
        cpu_solver/cpu_solver.h
        cpu_solver/cpu_main.h
        gpu_solver/gpu_main.cuh)

set_target_properties(SudokuSolverCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
