//
// Created by kamil on 14.11.2025.
//

#ifndef SUDOKUSOLVER_SUDOKU_H
#define SUDOKUSOLVER_SUDOKU_H
#include <cstdint>


class sudoku {
private:
    uint64_t rows[9];
public:
    explicit sudoku(uint64_t rows[9]);
    uint8_t getDigitAt(int row, int col);
};


#endif //SUDOKUSOLVER_SUDOKU_H