#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    extern __shared__ float sh[];

    float* Mds = sh;
    float* Nds = sh + TILE_WIDTH * TILE_WIDTH;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0.0f;
    int num_phases = (N + TILE_WIDTH - 1)/ TILE_WIDTH;

    for(int ph = 0; ph < num_phases; ++ph){
        int kbase = ph * TILE_WIDTH;
        if(Row < M && (kbase + tx) < N)
            Mds[ty * TILE_WIDTH + tx] = A[Row * N + (kbase + tx)];
        else
             Mds[ty * TILE_WIDTH + tx] = 0.0f;
        if((kbase + ty) < N && Col < K)
            Nds[ty * TILE_WIDTH + tx] = B[(kbase + ty) * K + Col];
        else
             Nds[ty * TILE_WIDTH + tx] = 0.0f;

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
        __syncthreads();
    }
    if(Row < M && Col < K)
        C[Row * K + Col] = Pvalue;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    size_t shmem = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, shmem>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
