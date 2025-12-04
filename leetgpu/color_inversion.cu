#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nPixels = width * height;
    if(tid < nPixels){
        int byteIndex = 4 * tid;
        #pragma unroll
        for(int i = 0; i < 3; ++i)
            image[byteIndex + i] = 255 - image[byteIndex + i];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
