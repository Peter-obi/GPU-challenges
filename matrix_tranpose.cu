#include <cuda_runtime.h>
#define TILE_WIDTH 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
	
	__shared__ float input_mat[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    if (Row < rows && Col < cols) 
    	input_mat[ty][tx] = input[Row * cols + Col];
    else
        input_mat[ty][tx] = 0.0f;
    __syncthreads();

    int tRow = bx * TILE_WIDTH + ty;
    int tCol = by * TILE_WIDTH + tx;

    if(tRow < cols && tCol < rows)
        output[tRow * rows + tCol] = input_mat[tx][ty];
    

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    size_t shmem = 2 * TILE_WIDTH * TILE_WIDTH + sizeof(float);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock, shmem>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
