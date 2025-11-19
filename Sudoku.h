//
// Created by kamil on 17.11.2025.
//

#ifndef SUDOKUSOLVERC_SUDOKU_H
#define SUDOKUSOLVERC_SUDOKU_H
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define SUDOKU_COUNT 100
#define SUDOKU_DIMENSION_SIZE 9
#define SUDOKU_BOARD_SIZE 81

typedef struct {
    uint64_t rows[SUDOKU_COUNT][6];
} Sudoku;

uint32_t getDigitAt(const Sudoku* sudoku, const uint32_t sudokuNo, uint32_t row, uint32_t col) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);

    if (row < 6)
        return sudoku->rows[sudokuNo][row] >> (20 + 4 * col) & 0xF;

    // row >= 6 - transform rows:
    // 6 -> 1
    // 7 -> 3
    // 8 -> 5
    row = 2 * row - 11;

    // Move to next row
    if (col >= 5) {
        row++;
        col -= 5;
    }

    return sudoku->rows[sudokuNo][row] >> (4 * col) & 0xF;
}

void setDigitAt(Sudoku* sudoku, const uint32_t sudokuNo, uint32_t row, uint32_t col, const uint32_t digit) {
    assert(row < SUDOKU_DIMENSION_SIZE && col < SUDOKU_DIMENSION_SIZE);
    assert(digit <= 9);

    if (row < 6) {
        sudoku->rows[sudokuNo][row] &= ~(0xF << (20 + 4 * col));
        sudoku->rows[sudokuNo][row] |= digit << (20 + 4 * col);
    }

    // row >= 6 - transformation as above
    row = 2 * row - 11;

    if (col >= 5) {
        row++;
        col -= 5;
    }

    sudoku->rows[sudokuNo][row] &= ~(0xF << (4 * col));
    sudoku->rows[sudokuNo][row] |= digit << (4 * col);
}

void printSudoku(const Sudoku* sudoku, const uint32_t sudokuNo) {
    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            unsigned char digit = getDigitAt(sudoku, sudokuNo, i, j) + '0';
            if (digit == '0')
                digit = '.';
            putc(digit, stdout);
            putc(' ', stdout);
        }
        putc('\n', stdout);
    }
}

#endif //SUDOKUSOLVERC_SUDOKU_H