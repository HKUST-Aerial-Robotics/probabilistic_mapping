#ifndef _filter_cost_h
#define _filter_cost_h

#include "cuda_utils.h"
#include "parameters.h"

void filter_cost(unsigned char *cost,
                 unsigned char *dep, size_t pitch_dep, float DEP_SAMPLE);

#endif
