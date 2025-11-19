cmake_minimum_required(VERSION 4.0)
project(SudokuSolverC C)

set(CMAKE_C_STANDARD 23)

add_executable(SudokuSolverC main.c
        sudoku.h
        args_parser.h)
