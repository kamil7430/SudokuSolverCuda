cmake_minimum_required(VERSION 4.0)
project(SudokuSolverCuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(SudokuSolverCuda main.cu
        sudoku.h
        args_parser.h)

set_target_properties(SudokuSolverCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
