//
// Created by kamil on 17.11.2025.
//

#ifndef SUDOKUSOLVERC_ARGS_PARSER_H
#define SUDOKUSOLVERC_ARGS_PARSER_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sudoku.h"

typedef struct {
    bool initialized;
    char method;
    int totalSudokuCount;
    int sudokusParsed;
    char buffer[SUDOKU_BOARD_SIZE];
    FILE* inputFile;
    FILE* outputFile;
} ArgsParser;

const char* validateAndParseArgs(const int argc, char** argv, ArgsParser* parser) {
    // Syntax: sudoku method count input_file output_file

    if (argc != 5)
        return "Arguments count is not valid - should be equal to 4!";

    if (strcmp(argv[1], "cpu") == 0)
        parser->method = 'c';
    else if (strcmp(argv[1], "gpu") == 0)
        parser->method = 'g';
    else
        return "Method should be \"cpu\" or \"gpu\"!";

    parser->totalSudokuCount = atoi(argv[2]);
    if (parser->totalSudokuCount <= 0)
        return "Sudoku count is not valid!";

    parser->inputFile = fopen(argv[3], "r");
    if (parser->inputFile == NULL)
        return "Failed to open input file for reading!";

    parser->outputFile = fopen(argv[4], "w");
    if (parser->outputFile == NULL)
        return "Failed to open output file for writing!";

    parser->initialized = true;
    return NULL;
}

int getNextSudoku(ArgsParser* parser, Sudoku* sudoku, const uint32_t sudokuNo) {
    if (!parser->initialized)
        return -1;

    if (parser->sudokusParsed == parser->totalSudokuCount) {
        return 0;
    }

    if (1 != fscanf(parser->inputFile, "%s\r\n", parser->buffer))
        return -2;

    for (int i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            const int val = parser->buffer[i * SUDOKU_DIMENSION_SIZE + j] - '0';
            if (val < 0 || val > 9)
                return -2;
            setDigitAt(sudoku, sudokuNo, i, j, val);
        }
    }

    parser->sudokusParsed++;
    return 1;
}

void printGetNextSudokuErrorMessage(const int err) {
    switch (err) {
        case -1:
            fprintf(stderr, "getNextSudoku error: ArgsParser is not initialized!\n");
            break;
        case -2:
            fprintf(stderr, "getNextSudoku error: Invalid input file format!\n");
            break;
        default:
            break;
    }
}

#endif //SUDOKUSOLVERC_ARGS_PARSER_H