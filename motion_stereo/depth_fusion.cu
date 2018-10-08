#include "depth_fusion.h"

__device__ __forceinline__ float normpdf(const float &x, const float &mu, const float &sigma_sq)
{
    return (expf(-(x-mu)*(x-mu) / (2.0f*sigma_sq))) * rsqrtf(2.0f*M_PI*sigma_sq);
}

__global__ void transformMap( float fx, float fy, float cx, float cy,
                              float r11, float r12, float r13,
                              float r21, float r22, float r23,
                              float r31, float r32, float r33,
                              float t1, float t2, float t3,
                              int* propogate_table,
                              float* alpha, float* beta, float* mu, float* sig )
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int width = WIDTH;
    const int height = HEIGHT;
    int index = y*width + x;

    if(x >= width || y >= height){
        return;
    }
    float a = *(alpha+index);
    if ( a < FLOAT_EPS ){
        return ;
    }
    float b = *(beta+index);
    if ( a/(a+b) < 0.4 ){
        return ;
    }
    float depth = *(mu+index) ;
    float px = (x-cx)/fx*depth;
    float py = (y-cy)/fy*depth;
    float pz = depth;
    float px2 = r11*px+r12*py+r13*pz + t1;
    float py2 = r21*px+r22*py+r23*pz + t2;
    float pz2 = r31*px+r32*py+r33*pz + t3;
    if ( pz2 < FLOAT_EPS ){
        return ;
    }
    int x2 = px2/pz2*fx + cx + 0.5 ;
    int y2 = py2/pz2*fy + cy + 0.5 ;
    if ( x2 < 0 || x2 >= width || y2 < 0 || y2 >= height ){
        return ;
    }
    *(mu + index) = pz2 ;
    *(sig + index) = *(sig + index) + pz2 * 0.05 ;

    int index2 = y2*width + x2 ;

    int *check_ptr = propogate_table + index2;
    int expect_i = 0;
    int actual_i;
    bool finish_job = false;
    int max_loop = 5;
    while(!finish_job && max_loop > 0)
    {
        max_loop--;
        actual_i = atomicCAS(check_ptr, expect_i, index);

        if ( actual_i != expect_i)
        {
            int now_x = actual_i % width;
            int now_y = actual_i / width;
            int index3 = now_y*width + now_x ;
            float now_d = *(mu + index3) ;
            if(now_d < pz2 ){
                finish_job = true;
            }
        }
        else {
            finish_job = true;
        }
        expect_i = actual_i;
    }
}

__global__ void hole_filling(int* propogate_table)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int width = WIDTH;
    const int height = HEIGHT;

    if( x <= 0 || x >= width-1 || y <= 0 || y >= height-1 ){
        return;
    }
    int index = y*width + x;
    const int transform_i = *(propogate_table + index);

    if(transform_i == 0){
        return;
    }

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            int k = (y+i)*width + x+j ;
            atomicCAS(propogate_table+k, 0, transform_i);
        }
    }
}

__global__ void depth_predict(
        float* pre_alpha, float* pre_beta, float* pre_mu, float* pre_sig,
        float* cur_alpha, float* cur_beta, float* cur_mu, float* cur_sig,
        int* propogate_table)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int width = WIDTH;
    const int height = HEIGHT;

    if(x >= width || y >= height){
        return;
    }
    int index = y*width + x ;

    const int transform_i = *(propogate_table+index);
    const int transform_x = transform_i % width;
    const int transform_y = transform_i / width;
    int pre_index = transform_y*width + transform_x;

    if(transform_i == 0){
        *(cur_alpha+index) = -1.0 ;
    }
    else{
        *(cur_alpha+index) = *(pre_alpha+pre_index) ;
        *(cur_beta+index) = *(pre_beta+pre_index) ;
        *(cur_mu+index) = *(pre_mu+pre_index) ;
        *(cur_sig+index) = *(pre_sig+pre_index) ;
    }
}

__global__ void depthUpdate(float* depth, float* alpha, float* beta, float*mu, float* sig)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int width = WIDTH;
    const int height = HEIGHT;
    int index = y*width + x;

    if(x >= width || y >= height)
        return;

    float depth_estimate =  *(depth + index) ;
    float uncertianity = depth_estimate * depth_estimate * 0.01;

//    *(alpha + index ) = 10.0f ;
//    *(beta + index ) = 10.0f ;
//    *(mu + index ) = depth_estimate ;
//    *(sig + index ) = 2500.0f ;

//    return ;


    if( *(alpha + index) < FLOAT_EPS )
    {
        if ( depth_estimate < FLOAT_EPS ){
            *(depth + index) = 0.0f;
        }
        else
        {
            *(alpha + index ) = 10.0f ;
            *(beta + index ) = 10.0f ;
            *(mu + index ) = depth_estimate ;
            *(sig + index ) = 2500.0f ;
        }
        return ;
    }

    if ( depth_estimate < FLOAT_EPS )
    {
        float a = *(alpha + index) ;
        float b = *(beta + index ) ;
        if( a/(a + b) > 0.6 ){
            *(depth + index) = *(mu + index);
        }
        else {
            *(depth + index) = 0 ;
        }
        return ;
    }

    //orieigin info
    float a = *(alpha + index) ;
    float b = *(beta + index ) ;
    float miu = *(mu + index ) ;
    float sigma_sq = *(sig + index ) ;

    float new_sq = uncertianity * sigma_sq / (uncertianity + sigma_sq);
    float new_miu = (depth_estimate * sigma_sq + miu * uncertianity) / (uncertianity + sigma_sq);
    float c1 = (a / (a+b)) * normpdf(depth_estimate, miu, uncertianity + sigma_sq);
    float c2 = (b / (a+b)) * 1 / 50.0f;

    const float norm_const = c1 + c2;
    c1 = c1 / norm_const;
    c2 = c2 / norm_const;
    const float f = c1 * ((a + 1.0f) / (a + b + 1.0f)) + c2 *(a / (a + b + 1.0f));
    const float e = c1 * (( (a + 1.0f)*(a + 2.0f)) / ((a + b + 1.0f) * (a + b + 2.0f))) +
            c2 *(a*(a + 1.0f) / ((a + b + 1.0f) * (a + b + 2.0f)));

    const float mu_prime = c1 * new_miu + c2 * miu;
    const float sigma_prime = c1 * (new_sq + new_miu * new_miu) + c2 * (sigma_sq + miu * miu) - mu_prime * mu_prime;
    const float a_prime = ( e - f ) / ( f - e/f );
    const float b_prime = a_prime * ( 1.0f - f ) / f;
    //const float4 updated = make_float4(a_prime, b_prime, mu_prime, sigma_prime);

    *(alpha + index) = a_prime ;
    *(beta + index) = b_prime ;
    *(mu + index) = mu_prime ;
    *(sig + index) = sigma_prime ;

    //  __syncthreads();
    if( a/(a + b) > 0.6 ){
        *(depth + index) = mu_prime;
    }
    else {
        *(depth + index) = 0.0f;
    }
}

//function define here

void depth_fuse(float fx, float fy, float cx, float cy,
                float r11, float r12, float r13,
                float r21, float r22, float r23,
                float r31, float r32, float r33,
                float t1, float t2, float t3,
                unsigned char* cur_depth, unsigned char* propogate_table,
                unsigned char* pre_alpha, unsigned char* pre_beta, unsigned char* pre_mu, unsigned char* pre_sig,
                unsigned char* cur_alpha, unsigned char* cur_beta, unsigned char* cur_mu, unsigned char* cur_sig )
{
    const int depth_width = WIDTH;
    const int depth_height = HEIGHT;

    dim3 fuse_block;
    dim3 fuse_grid;
    fuse_block.x = 16;
    fuse_block.y = 16;
    fuse_grid.x = (depth_width + fuse_block.x - 1) / fuse_block.x;
    fuse_grid.y = (depth_height + fuse_block.y - 1) / fuse_block.y;

    transformMap<<<fuse_grid, fuse_block>>>(fx, fy, cx, cy,
                                            r11, r12, r13,
                                            r21, r22, r23,
                                            r31, r32, r33,
                                            t1, t2, t3,
                                            reinterpret_cast<int *>(propogate_table),
                                            reinterpret_cast<float *>(pre_alpha),
                                            reinterpret_cast<float *>(pre_beta),
                                            reinterpret_cast<float *>(pre_mu),
                                            reinterpret_cast<float *>(pre_sig));
    cudaDeviceSynchronize();

    hole_filling<<<fuse_grid, fuse_block>>>(reinterpret_cast<int *>(propogate_table));
    cudaDeviceSynchronize();

    depth_predict<<<fuse_grid, fuse_block>>>(reinterpret_cast<float *>(pre_alpha),
                                             reinterpret_cast<float *>(pre_beta),
                                             reinterpret_cast<float *>(pre_mu),
                                             reinterpret_cast<float *>(pre_sig),
                                             reinterpret_cast<float *>(cur_alpha),
                                             reinterpret_cast<float *>(cur_beta),
                                             reinterpret_cast<float *>(cur_mu),
                                             reinterpret_cast<float *>(cur_sig),
                                             reinterpret_cast<int *>(propogate_table));
    cudaDeviceSynchronize();

    depthUpdate<<<fuse_grid, fuse_block>>>(reinterpret_cast<float *>(cur_depth),
                                           reinterpret_cast<float *>(cur_alpha),
                                           reinterpret_cast<float *>(cur_beta),
                                           reinterpret_cast<float *>(cur_mu),
                                           reinterpret_cast<float *>(cur_sig));
    cudaDeviceSynchronize();
}

