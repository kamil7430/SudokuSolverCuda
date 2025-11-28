//
// Created by kamil on 23.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_GPU_MAIN_CUH
#define SUDOKUSOLVERCUDA_GPU_MAIN_CUH

#include "gpu_solver.cuh"
#include <time.h>

void performFirstFewStepsOfRecursion(Sudoku* sudoku, Sudoku* outSudoku, int depth, int i, int j, int* offset) {
    // If depth of 3 is reached, finish the recursion and save the board
    if (depth == 3) {
        outSudoku[*offset] = *sudoku;
        (*offset)++;
        return;
    }

    // Look for a cell with the lowest number of possible digits
    int minPossibleDigits = 10;
    int row = 0, col = 0;
    for (i = 0; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (j = 0; j < SUDOKU_DIMENSION_SIZE; j++) {
            if (getDigitAt(sudoku, i, j) == 0) {
                const int possibleDigits = __builtin_popcount(getPossibleDigitsAt(sudoku, i, j));
                if (possibleDigits < minPossibleDigits) {
                    minPossibleDigits = possibleDigits;
                    row = i;
                    col = j;
                }
            }
        }
    }

    uint16_t digitsMask = getPossibleDigitsAt(sudoku, row, col);

    // Make recursive calls for all possible digits
    int digit = 0;
    while (digitsMask > 0) {
        const int shift = __builtin_ffs(digitsMask);
        digit += shift;
        digitsMask >>= shift;

        setDigitAndUpdateUsedDigits(sudoku, row, col, digit);
        performFirstFewStepsOfRecursion(sudoku, outSudoku, depth + 1, row, col + 1, offset);
        removeDigitAndUpdateUsedDigits(sudoku, row, col, digit);
    }
}

int gpu_main(Sudoku* sudokus, const int sudokuCount) {
    const clock_t preprocessingStart = clock();

    for (int i = 0; i < sudokuCount; i++) {
        cpuPreprocessSudoku(&sudokus[i]);
    }

    // Base: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Declare kernel parameters
    constexpr int threadsPerBlock = 256;
    const int blocks = sudokuCount;

    // Count the size of memory to allocate and declare pointers to device memory
    const unsigned int output_sudokus_size = sudokuCount * sizeof(Sudoku);
    const unsigned int input_sudokus_size = output_sudokus_size * threadsPerBlock;
    Sudoku* device_input_sudokus;
    Sudoku* device_output_sudokus;
    int* device_preprocessed_sudokus_count;

    // Split every sudoku board by taking first few steps of algorithm
    const unsigned int psc_size = sudokuCount * sizeof(int);
    Sudoku* preprocessed_sudokus = (Sudoku*)malloc(input_sudokus_size);
    int* preprocessed_sudokus_count = (int*)malloc(psc_size);
    for (int i = 0; i < sudokuCount; i++) {
        preprocessed_sudokus_count[i] = 0;
        performFirstFewStepsOfRecursion(&sudokus[i], preprocessed_sudokus + i * threadsPerBlock, 0, 0, 0, &preprocessed_sudokus_count[i]);
    }

    const clock_t preprocessingEnd = clock();
    printf("CPU preprocessing time: %f s\n", (float)(preprocessingEnd - preprocessingStart) / CLOCKS_PER_SEC);
    const clock_t copyingToDeviceStart = clock();

    // Allocate memory on the device
    err = cudaMalloc((void **)&device_input_sudokus, input_sudokus_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input sudokus (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void **)&device_output_sudokus, output_sudokus_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device output sudokus (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void **)&device_preprocessed_sudokus_count, psc_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device sudoku count (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy the host input sudokus in host memory to the device memory
    printf("Copy input data from the host memory to the CUDA device \n");
    err = cudaMemcpy(device_input_sudokus, preprocessed_sudokus, input_sudokus_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy sudokus from host to device (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    // Set output memory to zeros to handle contradictory sudoku boards
    err = cudaMemset(device_output_sudokus, 0, output_sudokus_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set output sudoku device memory (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(device_preprocessed_sudokus_count, preprocessed_sudokus_count, psc_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy sudokus from host to device (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    const clock_t copyingToDeviceEnd = clock();
    printf("Copying to GPU time: %f s\n", (float)(copyingToDeviceEnd - copyingToDeviceStart) / CLOCKS_PER_SEC);

    // Initialize CUDA timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch the CUDA Kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks, threadsPerBlock);
    oneBlockOneSudokuKernel<<<blocks, threadsPerBlock>>>(device_input_sudokus, device_output_sudokus, device_preprocessed_sudokus_count);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f s\n", milliseconds / 1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch oneBlockOneSudokuKernel kernel (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Execution failed (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy the device result sudokus in device memory to the host result sudokus in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(sudokus, device_output_sudokus, output_sudokus_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy results from device to host (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    // Free device global memory
    err = cudaFree(device_input_sudokus);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device input sudokus (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaFree(device_output_sudokus);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device output sudokus (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaFree(device_preprocessed_sudokus_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device output sudokus (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    // Free host memory
    free(preprocessed_sudokus);
    free(preprocessed_sudokus_count);

    printf("CUDA execution finished!\n");
    return 0;
}

#endif //SUDOKUSOLVERCUDA_GPU_MAIN_CUH