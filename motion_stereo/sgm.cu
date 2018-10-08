#include "sgm.h"

extern texture<float, cudaTextureType2D, cudaReadModeElementType> tex2Dleft;

template <int idx, int start, int dx, int dy, int n>
__global__ void sgm2(
    float *input,
    float *output,
    float pi1, float pi2, float tau_so, float sgm_q1, float sgm_q2)
{
    int xy[2];
    xy[idx] = start;
    xy[idx^1] = blockIdx.x;

    int x = xy[0], y = xy[1];
    int d = threadIdx.x;

    __shared__ float prev_Lrk[400], prev_Lri_min[400];
    __shared__ float cost[400], Lri_min[400];

    cost[d] = Lri_min[d] = input[INDEX(y, x, d)];
    __syncthreads();

    for (int i = DEP_CNT>>1; i>0; i>>=1)  // find the min Lr(p,d)
    {
        if (d < i && d + i < DEP_CNT && Lri_min[d + i] < Lri_min[d])
        {
            Lri_min[d] = Lri_min[d + i];
        }
        __syncthreads();
    }

    if (Lri_min[0] < 0.0f)  // if cost[][][] exist any disp out of cur_frame image, then set { \sum_r L_r } to INF
    {
        cost[d] = COST_INF;  // 0.0f
        output[INDEX(y, x, d)] = cost[d];
        prev_Lrk[d] = prev_Lri_min[d] = cost[d];
    }
    else
    {
        output[INDEX(y, x, d)] += cost[d];  // output is { \sum_r L_r }, so need to be sum up
        prev_Lrk[d] = prev_Lri_min[d] = cost[d];
    }
    xy[0] += dx;
    xy[1] += dy;

    for (int k = 1; k < n; k++, xy[0] += dx, xy[1] += dy)
    {
        x = xy[0];
        y = xy[1];

        cost[d] = Lri_min[d] = input[INDEX(y, x, d)];
        __syncthreads();

        for (int i = DEP_CNT>>1; i>0; i>>=1)
        {
            if (d < i && d + i < DEP_CNT && prev_Lri_min[d + i] < prev_Lri_min[d])
            {
                prev_Lri_min[d] = prev_Lri_min[d + i];
            }
            if (d < i && d + i < DEP_CNT && Lri_min[d + i] < Lri_min[d])
            {
                Lri_min[d] = Lri_min[d + i];
            }
            __syncthreads();
        }
        if (Lri_min[0] < 0.0f)
        {
            cost[d] = COST_INF;  // 0.0f
            __syncthreads();
        }

        // ---------------------------
        // judge ref_frame intensity gradient, if < tau_so, then reduce P1,P2
        float D1 = fabs(tex2D(tex2Dleft, x +0.5f, y +0.5f) - tex2D(tex2Dleft, x - dx +0.5f, y - dy +0.5f));

        float P1 = pi1, P2 = pi2;
        if (D1 < tau_so)    // q1,q2  (0,100),  if gradient is small, then P1,P2 is large, tend to smooth depth
        {
            P1 *= sgm_q1;
            P2 *= sgm_q2;
        }
        // ---------------------------

        float val = min(prev_Lrk[d], prev_Lri_min[0] + P2);
        if (d - 1 >= 0)
        {
            val = min(val, prev_Lrk[d - 1] + P1);
        }
        if (d + 1 < DEP_CNT)
        {
            val = min(val, prev_Lrk[d + 1] + P1);
        }

        // origin version is: val = cost[d] + val;
        // since in each thread [d], the prev_Lri_min[0] is the same, and prev_Lri_min[0] <= val in each [d],
        // so is ok to minus it to keep cost at a small float.
        val = cost[d] + val - prev_Lri_min[0];


        if (Lri_min[0] < 0.0f)  // if cost[][][] exist any disp out of cur_frame image, then set { \sum_r L_r } to INF
        {
            output[INDEX(y, x, d)] = COST_INF;  // 0.0f
        }
        else
        {
            output[INDEX(y, x, d)] += val;  // output is { \sum_r L_r }, so need to be sum up
        }

        __syncthreads();  // every thread should be done before move current val to previous
        prev_Lri_min[d] = prev_Lrk[d] = val;
    }
}

void sgm2(
    unsigned char *input,
    unsigned char *output)
{
    sgm2<0, 0, 1, 0, WIDTH> << <HEIGHT, DEP_CNT>>> (reinterpret_cast<float *>(input),
                                                    reinterpret_cast<float *>(output),
                                                    pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm2<0, WIDTH - 1, -1, 0, WIDTH> << <HEIGHT, DEP_CNT>>> (reinterpret_cast<float *>(input),
                                                             reinterpret_cast<float *>(output),
                                                             pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm2<1, 0, 0, 1, HEIGHT> << <WIDTH, DEP_CNT>>> (reinterpret_cast<float *>(input),
                                                    reinterpret_cast<float *>(output),
                                                    pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm2<1, HEIGHT - 1, 0, -1, HEIGHT> << <WIDTH, DEP_CNT>>> (reinterpret_cast<float *>(input),
                                                              reinterpret_cast<float *>(output),
                                                              pi1, pi2, tau_so, sgm_q1, sgm_q2);
    cudaDeviceSynchronize();
}
