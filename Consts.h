//
// Created by kamil on 14.11.2025.
//

#ifndef SUDOKUSOLVER_CONSTS_H
#define SUDOKUSOLVER_CONSTS_H

namespace Consts {
    constexpr int SUDOKU_DIMENSION_SIZE = 9;
    constexpr int SUDOKU_BOARD_SIZE = SUDOKU_DIMENSION_SIZE * SUDOKU_DIMENSION_SIZE;

    constexpr const char* USAGE_PATTERN = "Arguments syntax: <method> <count> <input> <output>\n"
                                          "method: cpu/gpu\n"
                                          "count: count of sudokus (integer) to load from input file\n"
                                          "input: input file name or path\n"
                                          "output: output file name or path";
}

#endif //SUDOKUSOLVER_CONSTS_H