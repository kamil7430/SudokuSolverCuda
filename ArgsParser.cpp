#include "ArgsParser.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

void ArgsParser::throwIfUninitialized() const {
    if (!initialized)
        throw std::logic_error("Parser not initialized!");
}

ArgsParser::ArgsParser(int argc, char **argv) {
    this->argc = argc;
    this->argv = argv;
}

ArgsParser::~ArgsParser() {
    // I leave the output file open and believe that it'll be closed by user.
    fclose(inputFile);
}

const char* ArgsParser::validateAndParseArgs() {
    // Syntax: sudoku method count input_file output_file

    if (argc != 5)
        return "Arguments count is not valid - should be equal to 4!";

    if (strcmp(argv[1], "cpu") == 0)
        method = 'c';
    else if (strcmp(argv[1], "gpu") == 0)
        method = 'g';
    else
        return R"(Method should be "cpu" or "gpu"!)";

    totalSudokuCount = atoi(argv[2]);
    if (totalSudokuCount <= 0)
        return "Sudoku count is not valid!";

    inputFile = fopen(argv[3], "r");
    if (inputFile == nullptr)
        return "Failed to open input file for reading!";

    outputFile = fopen(argv[4], "w");
    if (outputFile == nullptr)
        return "Failed to open output file for writing!";

    this->initialized = true;
    return nullptr;
}

Sudoku* ArgsParser::getNextSudoku() {
    throwIfUninitialized();

    auto throwInvalidFileFormat = []() -> void {
        throw std::logic_error("Invalid input file format!");
    };

    if (sudokusParsed == totalSudokuCount)
        return nullptr;

    if (1 != fscanf(inputFile, "%s\r\n", buffer))
        throwInvalidFileFormat();

    auto sudoku = new Sudoku();
    for (int i = 0; i < Consts::SUDOKU_DIMENSION_SIZE; i++) {
        for (int j = 0; j < Consts::SUDOKU_DIMENSION_SIZE; j++) {
            int val = buffer[i * Consts::SUDOKU_DIMENSION_SIZE + j] - '0';
            if (val < 0 || val > 9)
                throwInvalidFileFormat();
            sudoku->setDigitAt(i, j, val);
        }
    }

    sudokusParsed++;
    return sudoku;
}

FILE* ArgsParser::getOutputFile() const {
    throwIfUninitialized();
    return outputFile;
}

char ArgsParser::getMethod() const {
    throwIfUninitialized();
    return method;
}
