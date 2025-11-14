#include "ArgsParser.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

ArgsParser::ArgsParser(int argc, char **argv) {
    this->argc = argc;
    this->argv = argv;
    this->initialized = false;
    this->currentLine = 0;
    this->inputFile = nullptr;
}

ArgsParser::~ArgsParser() {
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

    sudokuCount = atoi(argv[2]);
    if (sudokuCount <= 0)
        return "Sudoku count is not valid!";

    inputFile = fopen(argv[3], "r");
    if (inputFile == nullptr)
        return "Failed to open input file for reading!";

    outputFile = fopen(argv[4], "w");
    if (outputFile == nullptr)
        return "Failed to open output file for writing!";

    this->initialized = true;
}

FILE* ArgsParser::getOutputFile() const {
    if (initialized)
        return outputFile;
    return nullptr;
}
