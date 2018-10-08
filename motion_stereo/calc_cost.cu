#include "calc_cost.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> tex2Dleft;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex2Dright;

int iDivUp(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void
ADCalcCostKernel(
    float *measurement_cnt,
    float r11, float r12, float r13,
    float r21, float r22, float r23,
    float r31, float r32, float r33,
    float t1, float t2, float t3,
    float *cost, float DEP_SAMPLE)
{
    const int tidx_base = blockDim.x * blockIdx.x;
    const int tidy = blockIdx.y;

    for (int k = 0, tidx = tidx_base; k < DEP_CNT; k++, tidx++)
        if (tidx >= 0 && tidx <= WIDTH - 1 && tidy >= 0 && tidy <= HEIGHT - 1)
        {
            // A patch totally has 9 points
            // (u,v,1)
            float x = r11 * tidx + r12 * tidy + r13 * 1.0f;  // h11 * \hat p
            float y = r21 * tidx + r22 * tidy + r23 * 1.0f;  // h12 * \hat p
            float z = r31 * tidx + r32 * tidy + r33 * 1.0f;  // h13 * \hat p

            // other 8 directions
            // up: (u,v-1,1)
            float xu = x - r12;
            float yu = y - r22;
            float zu = z - r32;

            // down: (u,v+1,1)
            float xd = x + r12;
            float yd = y + r22;
            float zd = z + r32;

            // left: (u-1,v,1)
            float xl = x - r11;
            float yl = y - r21;
            float zl = z - r31;

            // right: (u+1,v,1)
            float xr = x + r11;
            float yr = x + r21;
            float zr = x + r31;

            // up-left: (u-1,v-1,1)
            float xul = xu - r11;
            float yul = yu - r21;
            float zul = zu - r31;

            // down-right: (u+1,v+1,1)
            float xdr = xd + r11;
            float ydr = yd + r21;
            float zdr = zd + r31;

            // down-left: (u-1,v+1,1)
            float xld = xl + r12;
            float yld = yl + r22;
            float zld = zl + r32;

            // up-right: (u+1,v-1,1)
            float xru = xr - r12;
            float yru = xr - r22;
            float zru = xr - r32;

            int i = threadIdx.x;  // k^{th} inv depth value
            {
                float *cost_ptr = cost + INDEX(tidy, tidx, i);
                float *cnt_ptr = measurement_cnt + INDEX(tidy, tidx, i);
                float last_cost = *cost_ptr;
                float last_cnt = *cnt_ptr ;

                float tmp = 0.0f;
                float idep = i * DEP_SAMPLE;    // k * D. threadIdx.x begins at 0, so actually sample depth \in [*,INF]

                // 0: (u,v)
                {
                    float w = z + t3 * idep;
                    float u = (x + t1 * idep) / w;
                    float v = (y + t2 * idep) / w;

                    if (w < 0 || u < 0 || u > WIDTH - 1 || v < 0 || v > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx+0.5f, tidy+0.5f) - tex2D(tex2Dright, u+0.5f, v+0.5f));    // photometric error
                }

                // 1
                {
                    float wu = zu + t3 * idep;
                    float uu = (xu + t1 * idep) / wu;
                    float vu = (yu + t2 * idep) / wu;

                    if (wu < 0 || uu < 0 || uu > WIDTH - 1 || vu < 0 || vu > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx+0.5f, tidy + 1 +0.5f) - tex2D(tex2Dright, uu+0.5f, vu+0.5f));
                }

                // 2
                {
                    float wd = zd + t3 * idep;
                    float ud = (xd + t1 * idep) / wd;
                    float vd = (yd + t2 * idep) / wd;

                    if (wd < 0 || ud < 0 || ud > WIDTH - 1 || vd < 0 || vd > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx+0.5f, tidy - 1 +0.5f) - tex2D(tex2Dright, ud+0.5f, vd+0.5f));
                }

                // 3
                {
                    float wl = zl + t3 * idep;
                    float ul = (xl + t1 * idep) / wl;
                    float vl = (yl + t2 * idep) / wl;

                    if (wl < 0 || ul < 0 || ul > WIDTH - 1 || vl < 0 || vl > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx - 1 +0.5f, tidy+0.5f) - tex2D(tex2Dright, ul+0.5f, vl+0.5f));
                }

                // 4
                {
                    float wr = zr + t3 * idep;
                    float ur = (xr + t1 * idep) / wr;
                    float vr = (yr + t2 * idep) / wr;

                    if (wr < 0 || ur < 0 || ur > WIDTH - 1 || vr < 0 || vr > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx + 1 +0.5f, tidy+0.5f) - tex2D(tex2Dright, ur+0.5f, vr+0.5f));
                }

                // 5
                {

                    float wul = zul + t3 * idep;
                    float uul = (xul + t1 * idep) / wul;
                    float vul = (yul + t2 * idep) / wul;

                    if (wul < 0 || uul < 0 || uul > WIDTH - 1 || vul < 0 || vul > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx - 1 +0.5f, tidy - 1 +0.5f) - tex2D(tex2Dright, uul+0.5f, vul+0.5f));
                }

                // 6
                {
                    float wdr = zdr + t3 * idep;
                    float udr = (xdr + t1 * idep) / wdr;
                    float vdr = (ydr + t2 * idep) / wdr;

                    if (wdr < 0 || udr < 0 || udr > WIDTH - 1 || vdr < 0 || vdr > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx + 1+0.5f, tidy + 1+0.5f) - tex2D(tex2Dright, udr+0.5f, vdr+0.5f));
                }

                // 7
                {
                    float wld = zld + t3 * idep;
                    float uld = (xld + t1 * idep) / wld;
                    float vld = (yld + t2 * idep) / wld;

                    if (wld < 0 || uld < 0 || uld > WIDTH - 1 || vld < 0 || vld > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx - 1 +0.5f, tidy + 1 +0.5f) - tex2D(tex2Dright, uld+0.5f, vld+0.5f));
                }

                // 8
                {
                    float wru = zru + t3 * idep;
                    float uru = (xru + t1 * idep) / wru;
                    float vru = (yru + t2 * idep) / wru;

                    if (wru < 0 || uru < 0 || uru > WIDTH - 1 || vru < 0 || vru > HEIGHT - 1){
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx + 1 +0.5f, tidy - 1 +0.5f) - tex2D(tex2Dright, uru +0.5f, vru +0.5f));
                }
                if ( last_cost < 0 ){
                    *cost_ptr = tmp / 9.0f ;
                    *cnt_ptr = 1.0 ;
                }
                else{
                    *cost_ptr = (last_cnt*last_cost + tmp / 9.0f)/(last_cnt+1.0) ;
                    *cnt_ptr = last_cnt + 1.0;
                }
            }
        }
}

void ad_calc_cost(unsigned char *measurement_cnt,
    float r11, float r12, float r13,
    float r21, float r22, float r23,
    float r31, float r32, float r33,
    float t1, float t2, float t3,
    unsigned char *img_l, size_t pitch_img_l,
    unsigned char *img_r, size_t pitch_img_r,
    unsigned char *cost, float DEP_SAMPLE )
{
    checkCudaErrors(cudaUnbindTexture(tex2Dleft));
    checkCudaErrors(cudaUnbindTexture(tex2Dright));

    dim3 numThreads = dim3(DEP_CNT, 1, 1);
    dim3 numBlocks = dim3(iDivUp(WIDTH, DEP_CNT), HEIGHT);

    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<float>();

    tex2Dleft.addressMode[0] = cudaAddressModeBorder;
    tex2Dleft.addressMode[1] = cudaAddressModeBorder;
    tex2Dleft.filterMode = cudaFilterModeLinear;
    tex2Dleft.normalized = false;
    tex2Dright.addressMode[0] = cudaAddressModeBorder;
    tex2Dright.addressMode[1] = cudaAddressModeBorder;
    tex2Dright.filterMode = cudaFilterModeLinear;
    tex2Dright.normalized = false;

    size_t offset = 0;
    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dleft, reinterpret_cast<float *>(img_l), ca_desc0, WIDTH, HEIGHT, pitch_img_l));
    assert(offset == 0);

    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dright, reinterpret_cast<float *>(img_r), ca_desc1, WIDTH, HEIGHT, pitch_img_r));
    assert(offset == 0);

    ADCalcCostKernel << <numBlocks, numThreads>>> (reinterpret_cast<float *>(measurement_cnt),
                                                   r11, r12, r13,
                                                   r21, r22, r23,
                                                   r31, r32, r33,
                                                   t1, t2, t3,
                                                   reinterpret_cast<float *>(cost), DEP_SAMPLE );
    cudaDeviceSynchronize();
}
