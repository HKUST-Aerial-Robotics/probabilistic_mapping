#ifndef _calc_cost_h
#define _calc_cost_h

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "parameters.h"
#include "cuda_utils.h"

void ad_calc_cost(unsigned char *measurement_cnt,
    float r11, float r12, float r13,
    float r21, float r22, float r23,
    float r31, float r32, float r33,
    float t1, float t2, float t3,
    unsigned char *img_l, size_t pitch_img_l,
    unsigned char *img_r, size_t pitch_img_r,
    unsigned char *cost, float DEP_SAMPLE);

#endif
