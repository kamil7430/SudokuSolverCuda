# SudokuSolverCuda

A high-performance Sudoku solver written in C and CUDA, utilizing GPU parallelism and low-level bitwise optimizations.

## Building

To build the project, you need the **NVIDIA CUDA Toolkit** installed and added to your system path. And, of course, a CUDA-capable graphics card. You can build the project using the CMake file provided.

### Linux (GCC/Clang)
```bash
cmake -S . -B build
cmake --build build --config Release
```

### Windows (MSVC)
```bash
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

## Input

Every Sudoku board is represented as a 81-char line followed by CR+LF symbols (Windows new line convention). `0` stands for an empty cell and `1`-`9` digits stand for themselves. Look at the example:
```
000400560010506090000097300009020040600005000000370000502000000063000000000960800
```

## Usage

```
./build/SudokuSolverCuda method count input_file output_file
```
Where:
- `method` is `gpu` or `cpu`
- `count` is count of sudoku boards to load from file
