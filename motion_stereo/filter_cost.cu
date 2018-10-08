#include "filter_cost.h"

__global__ void
filterCostKernel(float *cost,
                 unsigned char *dep, size_t pitch_dep,
                 float var_scale, float DEP_SAMPLE)
{
    const int tidx = blockIdx.x;
    const int tidy = blockIdx.y;
    const int d = threadIdx.x;

    if (tidx >= 0 && tidx < WIDTH && tidx >= 0 && tidy < HEIGHT)
    {
        float *p_dep = (float *)(dep + tidy * pitch_dep) + tidx;

        __shared__ float c[DEP_CNT], c_min[DEP_CNT];
        __shared__ int c_idx[DEP_CNT];

        c[d] = c_min[d] = cost[INDEX(tidy, tidx, d)];
        c_idx[d] = d;
        __syncthreads();
        for (int i = DEP_CNT>>1; i>0; i>>=1) //  every time save min into half:  [0,32] -> [0,16] -> ... -> [0,0] -> min_cost
        {
            if (d < i && d + i < DEP_CNT && c_min[d + i] < c_min[d])
            {
                c_min[d] = c_min[d + i];
                c_idx[d] = c_idx[d + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            float min_cost = c_min[0];
            int min_idx = c_idx[0];

            // drop depth if exists 3 situations below:
            if ( (min_cost - 0.0) < FLOAT_EPS && (min_cost - 0.0) > -FLOAT_EPS )  // situation 1: can be considered as the non-overlap between cur_frame and ref_frame
                *p_dep = DEP_INF;
            else if (min_idx == 0 || min_idx == DEP_CNT - 1)  // situation 2
                *p_dep = 0.0;
            else if (c[min_idx - 1] + c[min_idx + 1] < 2 * min_cost * var_scale)  // situation 3
                *p_dep = 0.0;
            else
            {
                // y = a*x^2 + b*x + c
                float cost_pre = c[min_idx - 1];
                float cost_post = c[min_idx + 1];
                float a = cost_pre - 2.0f * min_cost + cost_post;  // 2*a
                float b = -cost_pre + cost_post;  // 4*a*x + 2*b
                float subpixel_idx = min_idx - b / (2.0f * a);    //  ans = - b/2a
                *p_dep = 1.0f / (subpixel_idx * DEP_SAMPLE);
            }
        }
    }
}

void filter_cost(
    unsigned char *cost,
    unsigned char *dep,
    size_t pitch_dep,
    float DEP_SAMPLE)
{
    dim3 numThreads = dim3(DEP_CNT, 1, 1);
    dim3 numBlocks = dim3(WIDTH, HEIGHT, 1);

    filterCostKernel << <numBlocks, numThreads>>> (reinterpret_cast<float *>(cost),
                                                   dep, pitch_dep,
                                                   var_scale, DEP_SAMPLE);
    cudaDeviceSynchronize();
}
