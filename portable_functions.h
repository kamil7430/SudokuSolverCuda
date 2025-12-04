//
// Created by kamil on 4.12.2025.
//

#ifndef SUDOKUSOLVERCUDA_PORTABLE_FUNCTIONS_H
#define SUDOKUSOLVERCUDA_PORTABLE_FUNCTIONS_H

#if defined(_MSC_VER) && !defined(__clang__)
    #include <intrin.h>
#endif

inline __host__ __device__ int portable_ffs(const int x) {
    #if defined(__CUDA_ARCH__)
        return __ffs(x);

    #elif defined(__GNUC__) || defined(__clang__)
        return __builtin_ffs(x);

    #elif defined(_MSC_VER)
        unsigned long index;
        if (_BitScanForward(&index, (unsigned long)x)) {
            return (int)(index + 1);
        }
        return 0;

    // Fallback - library function
    #else
        #include <strings.h>
        return ffs(x);
    #endif
}

inline __host__ __device__ int portable_popcount(const unsigned int x) {
    #if defined(__CUDA_ARCH__)
        return __popc(x);

    #elif defined(__GNUC__) || defined(__clang__)
        return __builtin_popcount(x);

    #elif defined(_MSC_VER)
        return (int)__popcnt(x);

    // Fallback (SWAR algorithm)
    #else
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F;
        return (x * 0x01010101) >> 24;
    #endif
}

#endif //SUDOKUSOLVERCUDA_PORTABLE_FUNCTIONS_H