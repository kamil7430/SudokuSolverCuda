//
// Created by kamil on 23.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_CPU_MAIN_H
#define SUDOKUSOLVERCUDA_CPU_MAIN_H

#include "cpu_iterative_solver.h"
#include "cpu_recursive_solver.h"

// Main function for solving sudokus on CPU
int cpu_main(Sudoku* sudoku) {
    static int sudokuNo = 1;
    // printf("### Solving sudoku %d ###\n", sudokuNo);
    int res = -1;

    if (cpuPreprocessSudoku(sudoku))
        puts("This sudoku board is invalid!\n");
    else {
        // int isSolved = 0;
        // const int result = cpuRecursiveBruteforceSolveSudoku(sudoku, &isSolved, 0, 0);
        const int result = cpuIterativeBruteforceSolveSudoku(sudoku);

        if (result > 0) {
            // puts("Sudoku solved!");
            res = 1;

            // if (const int result = validateSudokuSolution(sudoku)) {
            //     printf("Is solution valid: %d\n", result);
            //     // printSudoku(sudoku, stdout, 1);
            //     assert(result == 1);
            // }
        }
        else {
            puts("This sudoku is invalid!");
            // assert("Invalid sudoku!");
        }
    }

    sudokuNo++;
    return res;
}

#endif //SUDOKUSOLVERCUDA_CPU_MAIN_H