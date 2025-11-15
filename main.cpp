#include <iostream>

#include "ArgsParser.h"

int main(int argc, char** argv) {
    const auto parser = new ArgsParser(argc, argv);

    if (const auto errorMessage = parser->validateAndParseArgs()) {
        std::cerr << "Error: " << errorMessage << std::endl;
        std::cerr << Consts::USAGE_PATTERN << std::endl;
        delete parser;
        return 1;
    }
    const auto outputFile = parser->getOutputFile();

    try {
        Sudoku* sudoku;
        while ((sudoku = parser->getNextSudoku())) {
            // TODO: coś tam z sudoku
            sudoku->print();

            delete sudoku;
        }
    }
    catch (std::exception& e) {
        std::cerr << e.what();
    }

    fclose(outputFile);
    delete parser;
    return 0;
}