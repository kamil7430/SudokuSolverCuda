//
// Created by kamil on 20.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_CPU_RECURSIVE_SOLVER_H
#define SUDOKUSOLVERCUDA_CPU_RECURSIVE_SOLVER_H
#include "../sudoku.h"

int cpuPreprocessSudoku(Sudoku* sudoku) {
    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            const uint32_t digit = getDigitAt(sudoku, i, j);

            if (digit != 0) {
                const uint16_t possible = getPossibleDigitsAt(sudoku, i, j);
                if (possible >> (digit - 1) & ONE_BIT_MASK == 0)
                    return -1;

                updateUsedDigitsAt(sudoku, i, j, digit);
            }
        }
    }

    int restart = 0;
    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++)
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            if (getDigitAt(sudoku, i, j) != 0)
                continue;

            int digits = getPossibleDigitsAt(sudoku, i, j) & NINE_BIT_MASK;

            if (__builtin_popcount(digits) == 1) {
                setDigitAndUpdateUsedDigits(sudoku, i, j, __builtin_ffs(digits));
                restart = 1;
            }
        }

    if (restart)
        return cpuPreprocessSudoku(sudoku);

    return 0;
}

// Obsolete
int cpuRecursiveBruteforceSolveSudoku(Sudoku* sudoku, int* solved, int i, int j) {
    if (*solved)
        return 1;

    if (j >= SUDOKU_DIMENSION_SIZE) {
        i++;
        j = 0;
    }

    // We're out of sudoku bounds, so it is already solved
    if (i >= SUDOKU_DIMENSION_SIZE) {
        *solved = 1;
        return 1;
    }

    // Find next empty cell and bruteforce it
    for (; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (; j < SUDOKU_DIMENSION_SIZE; j++) {
            if (getDigitAt(sudoku, i, j) == 0) {
                uint16_t digitsMask = getPossibleDigitsAt(sudoku, i, j);

                int digit = 0;
                while (digitsMask > 0) {
                    const int shift = __builtin_ffs(digitsMask);
                    digit += shift;
                    digitsMask >>= shift;

                    setDigitAndUpdateUsedDigits(sudoku, i, j, digit);
                    cpuRecursiveBruteforceSolveSudoku(sudoku, solved, i, j + 1);
                    if (*solved)
                        return 1;
                    removeDigitAndUpdateUsedDigits(sudoku, i, j, digit);
                }

                return -1;
            }
        }
        j = 0;
    }

    // We have looked through whole board and all cells are filled - that's it!
    *solved = 1;
    return 1;
}

// Naive sudoku validation for development purposes
int validateSudokuSolution(const Sudoku* sudoku) {
    Sudoku validator = {};

    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++)
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            const uint32_t digit = getDigitAt(sudoku, i, j);

            if (digit < 1 || digit > 9)
                return -1;

            if (checkIfDigitIsPossible(&validator, i, j, digit) <= 0)
                return -1;

            setDigitAndUpdateUsedDigits(&validator, i, j, digit);
        }

    return 1;
}

#endif //SUDOKUSOLVERCUDA_CPU_RECURSIVE_SOLVER_H