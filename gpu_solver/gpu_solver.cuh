//
// Created by kamil on 25.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_GPU_SOLVER_CUH
#define SUDOKUSOLVERCUDA_GPU_SOLVER_CUH

#include "../sudoku.h"

__global__ void oneThreadOneSudokuKernel(Sudoku* sudokus, const int sudokuCount) {
    // Prepare data structures for bruteforce
    // empty_indices format: xxxxyyyy (8 bits):
    // xxxx - row (i)
    // yyyy - col (j)
    uint8_t empty_indices[SUDOKU_BOARD_SIZE] = {};
    uint8_t empty_count = 0; // Length of empty_indices array

    const int threadNumber = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadNumber >= sudokuCount)
        return;
    Sudoku sudoku = sudokus[threadNumber];

    // Find all empty cells
    for (uint8_t i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (uint8_t j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            if (getDigitAt(&sudoku, i, j) == 0) {
                empty_indices[empty_count] = (i << 4) | j;
                empty_count++;
            }
        }
    }

    // Perform iterative bruteforce
    int stack_idx = 0;
    while (stack_idx >= 0 && stack_idx < empty_count) {
        // Take one empty cell from "stack" array
        const uint8_t empty_cell_position = empty_indices[stack_idx];
        const uint8_t row = empty_cell_position >> 4;
        const uint8_t col = empty_cell_position & 0xF;

        // Clean up previous iteration (if occurred)
        const uint8_t previousDigit = getDigitAt(&sudoku, row, col);
        if (previousDigit != 0)
            removeDigitAndUpdateUsedDigits(&sudoku, row, col, previousDigit);

        // Bruteforce every digit
        uint16_t digitsMask = getPossibleDigitsAt(&sudoku, row, col);

        // If it's not first try for this cell, shift the possible digits mask
        digitsMask >>= previousDigit;

        if (digitsMask == 0) {
            // No possible digits - backtracking
            stack_idx--;
        }
        else {
            const int shift = __ffs(digitsMask);
            const int digit = previousDigit + shift;
            setDigitAndUpdateUsedDigits(&sudoku, row, col, digit);
            stack_idx++;
        }
    }

    // If sudoku is invalid, return an empty sudoku
    if (stack_idx != empty_count)
        memset(&sudoku, 0, sizeof(Sudoku));

    // Return solution
    sudokus[threadNumber] = sudoku;
}

#endif //SUDOKUSOLVERCUDA_GPU_SOLVER_CUH