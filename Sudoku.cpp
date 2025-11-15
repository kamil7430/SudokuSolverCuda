#include "Sudoku.h"

#include <cassert>
#include <cstdio>

Sudoku::Sudoku(const uint64_t rows[Consts::SUDOKU_DIMENSION_SIZE]) {
    for (int i = 0; i < Consts::SUDOKU_DIMENSION_SIZE; i++) {
        this->rows[i] = rows[i];
    }
}

uint8_t Sudoku::getDigitAt(unsigned int row, unsigned int col) const {
    assert(row < Consts::SUDOKU_DIMENSION_SIZE && col < Consts::SUDOKU_DIMENSION_SIZE);

    return (rows[row] & 0xF << (col << 2)) >> (col << 2);
}

void Sudoku::setDigitAt(unsigned int row, unsigned int col, unsigned int digit) {
    assert(row < Consts::SUDOKU_DIMENSION_SIZE && col < Consts::SUDOKU_DIMENSION_SIZE);
    assert(digit <= 9);

    rows[row] &= ~(0xF << (col << 2));
    rows[row] |= digit << (col << 2);
}

void Sudoku::print() const {
    for (int i = 0; i < Consts::SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < Consts::SUDOKU_DIMENSION_SIZE; j++) {
            unsigned char digit = getDigitAt(i, j) + '0';
            if (digit == '0')
                digit = '.';
            putc(digit, stdout);
            putc(' ', stdout);
        }
        putc('\n', stdout);
    }
}
