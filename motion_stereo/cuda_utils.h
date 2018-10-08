#ifndef _cuda_utils_h
#define _cuda_utils_h

#include <cstdio>

#include "parameters.h"

#ifdef __DRIVER_TYPES_H__
    #ifndef DEVICE_RESET
        #define DEVICE_RESET cudaDeviceReset();
    #endif
#else
    #ifndef DEVICE_RESET
        #define DEVICE_RESET
    #endif
#endif

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), "error", func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define INDEX(dim0, dim1, dim2) (((dim0)*WIDTH + (dim1)) * DEP_CNT + (dim2))

#endif

// Note:
// threadIdx.x is begin at 0
