#ifndef _depth_fusion_h
#define _depth_fusion_h

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "parameters.h"
#include "cuda_utils.h"

void depth_fuse(float fx, float fy, float cx, float cy,
                float r11, float r12, float r13,
                float r21, float r22, float r23,
                float r31, float r32, float r33,
                float t1, float t2, float t3,
                unsigned char *cur_depth, unsigned char *propogate_table,
                unsigned char *pre_alpha, unsigned char *pre_beta, unsigned char *pre_mu, unsigned char *pre_sig,
                unsigned char *cur_alpha, unsigned char *cur_beta, unsigned char *cur_mu, unsigned char *cur_sig );

#endif
