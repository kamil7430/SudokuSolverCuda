//
// Created by kamil on 23.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_GPU_MAIN_CUH
#define SUDOKUSOLVERCUDA_GPU_MAIN_CUH

#include "gpu_solver.cuh"

void performFirstThreeStepsOfRecursion(Sudoku* sudoku, Sudoku* outSudoku, int depth, int i, int j, int* offset) {
    if (depth == 3) {
        outSudoku[*offset] = *sudoku;
        (*offset)++;
        return;
    }

    for (; i < SUDOKU_DIMENSION_SIZE; i++) {
        for (; j < SUDOKU_DIMENSION_SIZE; j++) {
            if (getDigitAt(sudoku, i, j) == 0) {
                uint16_t digitsMask = getPossibleDigitsAt(sudoku, i, j);

                int digit = 0;
                while (digitsMask > 0) {
                    const int shift = __builtin_ffs(digitsMask);
                    digit += shift;
                    digitsMask >>= shift;

                    setDigitAndUpdateUsedDigits(sudoku, i, j, digit);
                    performFirstThreeStepsOfRecursion(sudoku, outSudoku, depth + 1, i, j + 1, offset);
                    removeDigitAndUpdateUsedDigits(sudoku, i, j, digit);
                }

                return;
            }
        }
        j = 0;
    }
}

int gpu_main(Sudoku* sudokus, const int sudokuCount) {
    for (int i = 0; i < sudokuCount; i++) {
        cpuPreprocessSudoku(&sudokus[i]);
    }

    // Base: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Declare kernel parameters
    constexpr int threadsPerBlock = 256;
    const int blocks = sudokuCount;

    // Allocate the device input sudokus
    const unsigned int output_sudokus_size = sudokuCount * sizeof(Sudoku);
    const unsigned int input_sudokus_size = output_sudokus_size * threadsPerBlock;
    Sudoku* device_input_sudokus;
    Sudoku* device_output_sudokus;
    int* device_preprocessed_sudokus_count;

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

    const int psc_size = sudokuCount * sizeof(int);
    err = cudaMalloc((void **)&device_preprocessed_sudokus_count, psc_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device sudoku count (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    Sudoku* preprocessed_sudokus = (Sudoku*)malloc(input_sudokus_size);
    int* preprocessed_sudokus_count = (int*)malloc(psc_size);
    for (int i = 0; i < sudokuCount; i++) {
        preprocessed_sudokus_count[i] = 0;
        performFirstThreeStepsOfRecursion(&sudokus[i], preprocessed_sudokus + i * threadsPerBlock, 0, 0, 0, &preprocessed_sudokus_count[i]);
        // printf("%d\n", preprocessed_sudokus_count[i]);
        // for (int j = 0; j < preprocessed_sudokus_count[i]; j++) {
        //     printSudoku(preprocessed_sudokus + i * threadsPerBlock + j, stdout, 1);
        //     putc('\n', stdout);
        // }
    }

    // Copy the host input sudokus in host memory to the device memory
    printf("Copy input data from the host memory to the CUDA device \n");
    err = cudaMemcpy(device_input_sudokus, preprocessed_sudokus, input_sudokus_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy sudokus from host to device (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(device_preprocessed_sudokus_count, preprocessed_sudokus_count, psc_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy sudokus from host to device (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch the CUDA Kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks, threadsPerBlock);
    oneBlockOneSudokuKernel<<<blocks, threadsPerBlock>>>(device_input_sudokus, sudokuCount, device_output_sudokus, device_preprocessed_sudokus_count);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch oneThreadOneSudokuKernel kernel (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Execution failed (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy the device result sudokus in device memory to the host result sudokus
    // in host memory.
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

    free(preprocessed_sudokus);
    free(preprocessed_sudokus_count);

    printf("CUDA execution finished!\n");
    return 0;
}

#endif //SUDOKUSOLVERCUDA_GPU_MAIN_CUH