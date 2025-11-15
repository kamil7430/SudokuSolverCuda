//
// Created by kamil on 14.11.2025.
//

#ifndef SUDOKUSOLVER_SUDOKU_H
#define SUDOKUSOLVER_SUDOKU_H
#include <cstdint>

#include "consts.h"


class Sudoku {
private:
    uint64_t rows[Consts::SUDOKU_DIMENSION_SIZE] = {};
public:
    explicit Sudoku(const uint64_t rows[Consts::SUDOKU_DIMENSION_SIZE]);
    Sudoku() = default;
    [[nodiscard]] uint8_t getDigitAt(unsigned int row, unsigned int col) const;
    void setDigitAt(unsigned int row, unsigned int col, unsigned int digit);
    void print() const;
};


#endif //SUDOKUSOLVER_SUDOKU_H