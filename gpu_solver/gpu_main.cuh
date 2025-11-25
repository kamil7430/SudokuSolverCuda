//
// Created by kamil on 23.11.2025.
//

#ifndef SUDOKUSOLVERCUDA_GPU_MAIN_CUH
#define SUDOKUSOLVERCUDA_GPU_MAIN_CUH

#include "gpu_solver.cuh"

int gpu_main(Sudoku* sudokus, const int sudokuCount) {
    // Base: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the device input sudokus
    const unsigned int sudokus_size = sudokuCount * sizeof(Sudoku);
    Sudoku *device_sudokus;

    err = cudaMalloc((void **)&device_sudokus, sudokus_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device sudokus (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input sudokus in host memory to the device memory
    err = cudaMemcpy(device_sudokus, sudokus, sudokus_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy sudokus from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    constexpr int threadsPerBlock = 1024;
    const int blocksPerGrid   = (sudokuCount + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    oneThreadOneSudokuKernel<<<blocksPerGrid, threadsPerBlock>>>(device_sudokus, sudokuCount);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch oneThreadOneSudokuKernel kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result sudokus in device memory to the host result sudokus
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(sudokus, device_sudokus, sudokus_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(device_sudokus);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device sudokus (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

#endif //SUDOKUSOLVERCUDA_GPU_MAIN_CUH