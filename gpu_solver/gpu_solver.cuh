//
// Created by kamil on 25.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_GPU_SOLVER_CUH
#define SUDOKUSOLVERCUDA_GPU_SOLVER_CUH

#include "../sudoku.h"
#include <cuda_runtime.h>

int __device__ gpuPreprocessSudoku(Sudoku* sudoku) {
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

            if (__popc(digits) == 1) {
                setDigitAndUpdateUsedDigits(sudoku, i, j, __ffs(digits));
                restart = 1;
            }
        }

    if (restart)
        return gpuPreprocessSudoku(sudoku);

    return 0;
}

__global__ void oneBlockOneSudokuKernel(Sudoku *sudokus, const int sudokuCount, Sudoku* outSudoku, int* preprocessedSudokusCount) {
    __shared__ volatile int foundSolution;

    if (threadIdx.x == 0) {
        foundSolution = 0;
    }
    __syncthreads();

    const int threadNumber = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIdx.x >= preprocessedSudokusCount[blockIdx.x])
        return;
    Sudoku sudoku = sudokus[threadNumber];

    // Prepare data structures for bruteforce
    // empty_indices format: xxxxyyyy (8 bits):
    // xxxx - row (i)
    // yyyy - col (j)
    uint8_t empty_indices[SUDOKU_BOARD_SIZE] = {};
    uint8_t empty_count = 0; // Length of empty_indices array

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
        // Leave if the solution was already found
        if (foundSolution)
            return;

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
        } else {
            const int shift = __ffs(digitsMask);
            const int digit = previousDigit + shift;
            setDigitAndUpdateUsedDigits(&sudoku, row, col, digit);
            stack_idx++;
        }
    }

    // If sudoku is invalid, return an empty sudoku
    if (stack_idx == empty_count) {
        if (foundSolution)
            return;
        foundSolution = 1;

        outSudoku[blockIdx.x] = sudoku;
    }

    // printf("Thread #%d finished!\n", threadNumber);
}

#endif //SUDOKUSOLVERCUDA_GPU_SOLVER_CUH
