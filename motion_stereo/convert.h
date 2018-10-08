#ifndef _convert_h
#define _convert_h

#include <cuda_runtime.h>

#include "parameters.h"
#include "cuda_utils.h"

void convert(unsigned char *input, unsigned char *output, bool flag, size_t pitch_dep);

#endif
