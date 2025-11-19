#include <stdio.h>

#include "args_parser.h"

#define USAGE_PATTERN "Arguments syntax: <method> <count> <input> <output>\n"\
                      "method: cpu/gpu\n"\
                      "count: count of sudokus (integer) to load from input file\n"\
                      "input: input file name or path\n"\
                      "output: output file name or path"

int main(const int argc, char** argv) {
    ArgsParser parser;

    const char* errorMessage = validateAndParseArgs(argc, argv, &parser);
    if (errorMessage) {
        fprintf(stderr, "Error: %s\n%s", errorMessage, USAGE_PATTERN);
        return 1;
    }
    FILE* outputFile = parser.outputFile;

    Sudoku sudoku;
    int err;
    int sudokuNo = 0;
    while ((err = getNextSudoku(&parser, &sudoku, sudokuNo)) > 0) {
        // TODO: coś tam z sudoku
        printSudoku(&sudoku, sudokuNo);
        sudokuNo++;
    }
    printGetNextSudokuErrorMessage(err);

    fclose(parser.inputFile);
    fclose(outputFile);
    return 0;
}