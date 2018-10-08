#include "convert.h"


__global__ void convertData(float *input, float* output, bool flag, size_t pitch_dep)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int width = WIDTH;
    const int height = HEIGHT;
    int indexInput ;
    int indexOuput ;
    if( x >= width || y >= height ){
        return;
    }
    if ( flag ){
        indexInput = y*width + x;
        indexOuput = y*pitch_dep + x;
    }
    else {
        indexInput = y*pitch_dep + x;
        indexOuput = y*width + x;
    }
    *(output + indexOuput) = *(input + indexInput) ;
}

void convert(
    unsigned char *input,
    unsigned char *output,
    bool flag, size_t pitch_dep)
{
    const int depth_width = WIDTH;
    const int depth_height = HEIGHT;

    dim3 fuse_block;
    dim3 fuse_grid;
    fuse_block.x = 16;
    fuse_block.y = 16;
    fuse_grid.x = (depth_width + fuse_block.x - 1) / fuse_block.x;
    fuse_grid.y = (depth_height + fuse_block.y - 1) / fuse_block.y;

    convertData<<<fuse_grid, fuse_block>>>(reinterpret_cast<float *>(input),
                                           reinterpret_cast<float *>(output),
                                           flag, pitch_dep);
    cudaDeviceSynchronize();
}
