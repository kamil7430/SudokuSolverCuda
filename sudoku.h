//
// Created by kamil on 17.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_SUDOKU_H
#define SUDOKUSOLVERCUDA_SUDOKU_H
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define SUDOKU_COUNT 100
#define SUDOKU_DIMENSION_SIZE 9
#define SUDOKU_BOARD_SIZE 81
#define SUDOKUS_PER_STRUCT 512

#define ONE_BIT_MASK (uint16_t)0x1
#define FOUR_BIT_MASK 0xFULL
#define NINE_BIT_MASK (uint16_t)0x01FF

typedef struct {
    uint64_t rows[SUDOKUS_PER_STRUCT][6];
    uint16_t usedDigitsInRow[SUDOKUS_PER_STRUCT][9];
    uint16_t usedDigitsInCol[SUDOKUS_PER_STRUCT][9];
    uint16_t usedDigitsInBox[SUDOKUS_PER_STRUCT][3][3];
} Sudoku;

inline __host__ __device__ uint16_t getUsedDigitsInBoxConst(const Sudoku* sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    return sudoku->usedDigitsInBox[sudokuNo][row / 3][col / 3];
}

inline __host__ __device__ uint16_t* getUsedDigitsInBoxPointer(Sudoku* sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    return &sudoku->usedDigitsInBox[sudokuNo][row / 3][col / 3];
}

__host__ __device__ uint32_t getDigitAt(const Sudoku* sudoku, const uint32_t sudokuNo, uint32_t row, uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    if (row < 6)
        return sudoku->rows[sudokuNo][row] >> (20 + 4 * col) & FOUR_BIT_MASK;

    // row >= 6 - transform rows:
    // 6 -> 0
    // 7 -> 2
    // 8 -> 4
    row = 2 * row - 12;

    // Move to next row
    if (col >= 5) {
        row++;
        col -= 5;
    }

    return sudoku->rows[sudokuNo][row] >> (4 * col) & FOUR_BIT_MASK;
}

__host__ __device__ void setDigitAt(Sudoku* sudoku, const uint32_t sudokuNo, uint32_t row, uint32_t col, const uint32_t digit) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);
    assert(digit <= 9);

    if (row < 6) {
        sudoku->rows[sudokuNo][row] &= ~(FOUR_BIT_MASK << (20 + 4 * col));
        sudoku->rows[sudokuNo][row] |= (uint64_t)digit << (20 + 4 * col);
        return;
    }

    // row >= 6 - transformation as above
    row = 2 * row - 12;

    if (col >= 5) {
        row++;
        col -= 5;
    }

    sudoku->rows[sudokuNo][row] &= ~(FOUR_BIT_MASK << (4 * col));
    sudoku->rows[sudokuNo][row] |= (uint64_t)digit << (4 * col);
}

__host__ __device__ uint16_t getPossibleDigitsAt(const Sudoku* sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    return ~(sudoku->usedDigitsInRow[sudokuNo][row] | sudoku->usedDigitsInCol[sudokuNo][col] | getUsedDigitsInBoxConst(sudoku, sudokuNo, row, col)) & NINE_BIT_MASK;
}

// Returns non-zero value if digit is possible, zero otherwise.
__host__ __device__ int checkIfDigitIsPossible(const Sudoku* sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col, const uint32_t digit) {
    assert(digit >= 1 && digit <= 9);

    const uint16_t possible = getPossibleDigitsAt(sudoku, sudokuNo, row, col);
    return possible & ONE_BIT_MASK << (digit - 1);
}

__host__ __device__ void updateUsedDigitsAt(Sudoku* sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col, const uint32_t digit) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);
    assert(digit >= 1 && digit <= 9);

    const uint16_t mask = ONE_BIT_MASK << (digit - 1);
    sudoku->usedDigitsInRow[sudokuNo][row] |= mask;
    sudoku->usedDigitsInCol[sudokuNo][col] |= mask;
    *getUsedDigitsInBoxPointer(sudoku, sudokuNo, row, col) |= mask;
}

__host__ __device__ void removeFromUsedDigitsAt(Sudoku* sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col, const uint32_t digit) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);
    assert(digit >= 1 && digit <= 9);

    const uint16_t mask = ~(ONE_BIT_MASK << (digit - 1));
    sudoku->usedDigitsInRow[sudokuNo][row] &= mask;
    sudoku->usedDigitsInCol[sudokuNo][col] &= mask;
    *getUsedDigitsInBoxPointer(sudoku, sudokuNo, row, col) &= mask;
}

__host__ __device__ void setDigitAndUpdateUsedDigits(Sudoku* sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col, const uint32_t digit) {
    setDigitAt(sudoku, sudokuNo, row, col, digit);
    updateUsedDigitsAt(sudoku, sudokuNo, row, col, digit);
}

__host__ __device__ void removeDigitAndUpdateUsedDigits(Sudoku *sudoku, const uint32_t sudokuNo, const uint32_t row, const uint32_t col, const uint32_t digit) {
    setDigitAt(sudoku, sudokuNo, row, col, 0);
    removeFromUsedDigitsAt(sudoku, sudokuNo, row, col, digit);
}

void printSudoku(const Sudoku* sudoku, const uint32_t sudokuNo, FILE* stream, int pretty) {
    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            unsigned char digit = getDigitAt(sudoku, sudokuNo, i, j) + '0';
            if (pretty && digit == '0')
                digit = '.';
            putc(digit, stream);
            if (pretty)
                putc(' ', stream);
        }
        if (pretty)
            putc('\n', stream);
    }
    if (!pretty)
        fputs("\r\n", stream);
}

__host__ __device__ void copySudoku(Sudoku* dst, const uint32_t dstNo, Sudoku* src, const uint32_t srcNo) {
    memcpy(dst->rows[dstNo], src->rows[srcNo], 6 * sizeof(uint64_t));
    memcpy(dst->usedDigitsInRow[dstNo], src->usedDigitsInRow[srcNo], 9 * sizeof(uint16_t));
    memcpy(dst->usedDigitsInCol[dstNo], src->usedDigitsInCol[srcNo], 9 * sizeof(uint16_t));
    memcpy(dst->usedDigitsInBox[dstNo][0], src->usedDigitsInBox[srcNo][0], 3 * sizeof(uint16_t));
    memcpy(dst->usedDigitsInBox[dstNo][1], src->usedDigitsInBox[srcNo][1], 3 * sizeof(uint16_t));
    memcpy(dst->usedDigitsInBox[dstNo][2], src->usedDigitsInBox[srcNo][2], 3 * sizeof(uint16_t));
}

#endif //SUDOKUSOLVERCUDA_SUDOKU_H
