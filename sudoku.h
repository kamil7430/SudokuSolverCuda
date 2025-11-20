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

#define ONE_BIT_MASK (uint16_t)0x1
#define FOUR_BIT_MASK 0xFULL
#define NINE_BIT_MASK (uint16_t)0x01FF

typedef struct {
    uint64_t rows[6];
    uint16_t usedDigitsInRow[9];
    uint16_t usedDigitsInCol[9];
    uint16_t usedDigitsInBox[3][3];
} Sudoku;

inline uint16_t getUsedDigitsInBoxConst(const Sudoku* sudoku, const uint32_t row, const uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    return sudoku->usedDigitsInBox[row / 3][col / 3];
}

inline uint16_t* getUsedDigitsInBoxPointer(Sudoku* sudoku, const uint32_t row, const uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    return &sudoku->usedDigitsInBox[row / 3][col / 3];
}

uint32_t getDigitAt(const Sudoku* sudoku, uint32_t row, uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    if (row < 6)
        return sudoku->rows[row] >> (20 + 4 * col) & FOUR_BIT_MASK;

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

    return sudoku->rows[row] >> (4 * col) & FOUR_BIT_MASK;
}

void setDigitAt(Sudoku* sudoku, uint32_t row, uint32_t col, const uint32_t digit) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);
    assert(digit <= 9);

    if (row < 6) {
        sudoku->rows[row] &= ~(FOUR_BIT_MASK << (20 + 4 * col));
        sudoku->rows[row] |= (uint64_t)digit << (20 + 4 * col);
        return;
    }

    // row >= 6 - transformation as above
    row = 2 * row - 12;

    if (col >= 5) {
        row++;
        col -= 5;
    }

    sudoku->rows[row] &= ~(FOUR_BIT_MASK << (4 * col));
    sudoku->rows[row] |= (uint64_t)digit << (4 * col);
}

uint16_t getPossibleDigitsAt(const Sudoku* sudoku, const uint32_t row, const uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    return ~(sudoku->usedDigitsInRow[row] | sudoku->usedDigitsInCol[col] | getUsedDigitsInBoxConst(sudoku, row, col)) & NINE_BIT_MASK;
}

void updateUsedDigitsAt(Sudoku* sudoku, const uint32_t row, const uint32_t col, const uint32_t digit) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);
    assert(digit >= 1 && digit <= 9);

    const uint16_t mask = ONE_BIT_MASK << (digit - 1);
    sudoku->usedDigitsInRow[row] |= mask;
    sudoku->usedDigitsInCol[col] |= mask;
    *getUsedDigitsInBoxPointer(sudoku, row, col) |= mask;
}

void setDigitAndUpdateUsedDigits(Sudoku* sudoku, const uint32_t row, const uint32_t col, const uint32_t digit) {
    setDigitAt(sudoku, row, col, digit);
    updateUsedDigitsAt(sudoku, row, col, digit);
}

void printSudoku(const Sudoku* sudoku) {
    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            unsigned char digit = getDigitAt(sudoku, i, j) + '0';
            if (digit == '0')
                digit = '.';
            putc(digit, stdout);
            putc(' ', stdout);
        }
        putc('\n', stdout);
    }
}

#endif //SUDOKUSOLVERCUDA_SUDOKU_H
