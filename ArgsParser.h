//
// Created by kamil on 14.11.2025.
//

#ifndef SUDOKUSOLVER_ARGSPARSER_H
#define SUDOKUSOLVER_ARGSPARSER_H

#include <cwchar>
#include "sudoku.h"


class ArgsParser {
private:
    int argc;
    char** argv;
    bool initialized;
    char method;
    int sudokuCount;
    int currentLine;
    FILE* inputFile;
    FILE* outputFile;
public:
    ArgsParser(int argc, char** argv);
    ~ArgsParser();
    [[nodiscard]] const char* validateAndParseArgs();
    [[nodiscard]] sudoku* getNextSudoku();
    [[nodiscard]] FILE* getOutputFile() const;
};


#endif //SUDOKUSOLVER_ARGSPARSER_H