//
// Created by kamil on 25.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_CPU_ITERATIVE_SOLVER_H
#define SUDOKUSOLVERCUDA_CPU_ITERATIVE_SOLVER_H
#include "../portable_functions.h"
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
                        const int possibleDigits = portable_popcount(getPossibleDigitsAt(sudoku, i, j));
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
        const uint8_t previousDigit = getDigitAt(sudoku, row, col);
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
            const int shift = portable_ffs(digitsMask);
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

int cpuPreprocessSudoku(Sudoku* sudoku) {
    // Update used digits in Sudoku struct
    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            const uint32_t digit = getDigitAt(sudoku, i, j);

            if (digit != 0) {
                const uint16_t possible = getPossibleDigitsAt(sudoku, i, j);

                // Looking for a contradiction
                if (possible >> (digit - 1) & ONE_BIT_MASK == 0)
                    return -1;

                updateUsedDigitsAt(sudoku, i, j, digit);
            }
        }
    }

    // Filling cells for which there is only one possible digit
    int restart = 0;
    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++)
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            if (getDigitAt(sudoku, i, j) != 0)
                continue;

            const int digits = getPossibleDigitsAt(sudoku, i, j) & NINE_BIT_MASK;

            if (portable_popcount(digits) == 1) {
                setDigitAndUpdateUsedDigits(sudoku, i, j, portable_ffs(digits));
                restart = 1;
            }
        }

    if (restart)
        return cpuPreprocessSudoku(sudoku);

    return 0;
}

// Naive sudoku validation for development and testing purposes
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

#endif //SUDOKUSOLVERCUDA_CPU_ITERATIVE_SOLVER_H