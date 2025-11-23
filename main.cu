#include <stdio.h>

#include "args_parser.h"
#include "cpu_solver/cpu_main.h"
#include "gpu_solver/gpu_main.cuh"

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
    constexpr Sudoku emptySudoku = {};
    int err;
    while ((err = getNextSudoku(&parser, &sudoku)) > 0) {
        int result = -1;

        switch (parser.method) {
            case 'c':
                result = cpu_main(&sudoku);
                break;
            case 'g':
                result = gpu_main(&sudoku);
                break;
            default:
                fputs("Unknown method type!", stderr);
                break;
        }

        if (result > 0)
            printSudoku(&sudoku, outputFile, 0);
        else
            // If the sudoku is invalid, save an empty output (consisting of zeros)
            printSudoku(&emptySudoku, outputFile, 0);
    }
    printGetNextSudokuErrorMessage(err);

    fclose(parser.inputFile);
    fclose(outputFile);
    return 0;
}