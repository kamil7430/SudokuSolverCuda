//
// Created by kamil on 23.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_CPU_MAIN_H
#define SUDOKUSOLVERCUDA_CPU_MAIN_H

#include "cpu_iterative_solver.h"
#include "cpu_recursive_solver.h"

// Main function for solving sudokus on CPU
void cpu_main(Sudoku* sudokus, const int sudokuCount) {
    for (int i = 0; i < sudokuCount; i++) {
        Sudoku* sudoku = &sudokus[i / SUDOKUS_PER_STRUCT];
        const int sudokuNo = i % SUDOKUS_PER_STRUCT;

        printf("### Solving sudoku %d/%03d ###\n", i / SUDOKUS_PER_STRUCT, sudokuNo);

        if (cpuPreprocessSudoku(sudoku, sudokuNo))
            puts("This sudoku board is invalid!\n");
        else {
            const int result = cpuIterativeBruteforceSolveSudoku(sudoku, sudokuNo);

            if (result > 0) {
                puts("Sudoku solved!");

                // if (const int validationResult = validateSudokuSolution(sudoku, sudokuNo)) {
                //     printf("Is solution valid: %d\n", validationResult);
                //     assert(validationResult == 1);
                // }
            }
            else {
                memset(sudoku->rows[sudokuNo], 0, 6 * sizeof(uint64_t));
                puts("This sudoku is invalid!");
            }
        }

        // printSudoku(sudoku, sudokuNo, stdout, 1);
        // puts("");
    }
}

#endif //SUDOKUSOLVERCUDA_CPU_MAIN_H