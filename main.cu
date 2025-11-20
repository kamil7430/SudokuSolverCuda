#include <stdio.h>

#include "args_parser.h"
#include "cpu_solver/cpu_solver.h"

#define USAGE_PATTERN "Arguments syntax: <method> <count> <input> <output>\n"\
                      "method: cpu/gpu\n"\
                      "count: count of sudokus (integer) to load from input file\n"\
                      "input: input file name or path\n"\
                      "output: output file name or path"

int main(const int argc, char** argv) {
    ArgsParser parser = {};

    const char* errorMessage = validateAndParseArgs(argc, argv, &parser);
    if (errorMessage) {
        fprintf(stderr, "Error: %s\n%s", errorMessage, USAGE_PATTERN);
        return 1;
    }
    FILE* outputFile = parser.outputFile;

    Sudoku sudoku;
    int err;
    while ((err = getNextSudoku(&parser, &sudoku)) > 0) {
        // TODO: coś tam z sudoku
        puts("-----\n");
        printSudoku(&sudoku);

        if (cpuPreprocessSudoku(&sudoku))
            puts("Sudoku jest sprzeczne!\n");

        int solved = 0;
        if (int result = cpuBruteforceSolveSudoku(&sudoku, &solved, 0, 0)) {
            printf("Wynik działania: %d\n", result);
        }

        puts("-----\n");
        printSudoku(&sudoku);
    }
    printGetNextSudokuErrorMessage(err);

    fclose(parser.inputFile);
    fclose(outputFile);
    return 0;
}