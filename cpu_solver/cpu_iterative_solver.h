//
// Created by kamil on 25.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_CPU_ITERATIVE_SOLVER_H
#define SUDOKUSOLVERCUDA_CPU_ITERATIVE_SOLVER_H
#include "../sudoku.h"

int cpuIterativeBruteforceSolveSudoku(Sudoku* sudoku) {
    // Prepare data structures for bruteforce
    // empty_indices format: xxxxyyyy (8 bits):
    // xxxx - row (i)
    // yyyy - col (j)
    uint8_t empty_indices[SUDOKU_BOARD_SIZE];
    memset(empty_indices, 0xFF, SUDOKU_BOARD_SIZE);

    // Perform iterative bruteforce
    int stack_idx = 0;
    do {
        uint8_t row = 0, col = 0;
        if (empty_indices[stack_idx] == 0xFF) {
            // Find a cell with the lowest possible digits count
            uint8_t minPossibleDigits = 10;
            for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
                for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
                    if (getDigitAt(sudoku, i, j) == 0) {
                        int possibleDigits = __builtin_popcount(getPossibleDigitsAt(sudoku, i, j));
                        if (possibleDigits < minPossibleDigits) {
                            minPossibleDigits = possibleDigits;
                            row = i;
                            col = j;
                        }
                    }
                }
            }
            if (minPossibleDigits >= 10)
                break;

            // Push it on the stack
            empty_indices[stack_idx] = (row << 4) | col;
        }
        else {
            row = empty_indices[stack_idx] >> 4;
            col = empty_indices[stack_idx] & 0xF;
        }

        // Clean up previous iteration (if occurred)
        uint8_t previousDigit = getDigitAt(sudoku, row, col);
        if (previousDigit != 0)
            removeDigitAndUpdateUsedDigits(sudoku, row, col, previousDigit);

        // Bruteforce every digit
        uint16_t digitsMask = getPossibleDigitsAt(sudoku, row, col);

        // If it's not first try for this cell, shift the possible digits mask
        digitsMask >>= previousDigit;

        if (digitsMask == 0) {
            // No possible digits - backtracking
            empty_indices[stack_idx] = 0xFF;
            stack_idx--;
        }
        else {
            const int shift = __builtin_ffs(digitsMask);
            const int digit = previousDigit + shift;
            setDigitAndUpdateUsedDigits(sudoku, row, col, digit);
            stack_idx++;
            //printSudoku(sudoku, stdout, 1);
        }
    } while (stack_idx >= 0);

    // Check if the solution was found
    if (stack_idx > 0) {
        return 1;
    }

    return -1;
}

#endif //SUDOKUSOLVERCUDA_CPU_ITERATIVE_SOLVER_H