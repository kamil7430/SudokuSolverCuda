cmake_minimum_required(VERSION 4.0)
project(SudokuSolver)

set(CMAKE_CXX_STANDARD 20)

add_executable(SudokuSolver main.cpp
        ArgsParser.h
        global_defines.h
        sudoku.h
        sudoku.cpp
        ArgsParser.cpp)
