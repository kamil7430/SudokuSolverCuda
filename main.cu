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

    const unsigned int sudokus_size = parser.totalSudokuCount * sizeof(Sudoku);
    Sudoku* sudokus = (Sudoku*)malloc(sudokus_size);
    if (sudokus == NULL) {
        fputs("Failed to allocate host memory with malloc!\n", stderr);
        fclose(parser.inputFile);
        fclose(outputFile);
        return 1;
    }

    memset(sudokus, 0, sudokus_size);

    int err, id = 0;
    while ((err = getNextSudoku(&parser, &sudokus[id])) > 0)
        id++;
    if (err < 0) {
        printGetNextSudokuErrorMessage(err);
        fclose(parser.inputFile);
        fclose(outputFile);
        return 1;
    }

    if (parser.method == 'c') {
        const clock_t startCpuSolving = clock();

        cpu_main(sudokus, parser.totalSudokuCount);

        const clock_t stopCpuSolving = clock();
        printf("CPU solving time: %f s\n", (float)(stopCpuSolving - startCpuSolving) / CLOCKS_PER_SEC);
    }
    else if (parser.method == 'g') {
        if (gpu_main(sudokus, parser.totalSudokuCount) <= 0) {
            for (int i = 0; i < parser.totalSudokuCount; i++) {
                if (validateSudokuSolution(&sudokus[i]) < 0) {
                    printf("Invalid solution for %d!\n", i);
                    printSudoku(&sudokus[i], stdout, 1);
                }
            }
        }
    }
    else {
        puts("Invalid parser method!\n");
    }

    free(sudokus);
    fclose(parser.inputFile);
    fclose(outputFile);
    return 0;
}