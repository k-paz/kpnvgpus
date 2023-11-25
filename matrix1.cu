#include <stdio.h>

#define N 10000000

__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{ 
    // Calculate the row & column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    if( row < m && col < k ) 
    {
        for(int j = 0; j < n; j++) 
        {
            sum += a[row * n + j] * b[j * k + col];
        }
        // Each thread computes one element of the block sub-matrix
        c[row * k + col] = sum;
	printf("%d\n", sum);
    }
} 

__global__ void helloFromGPU (void) {
    printf("and hello from A30 GPU!\n");
}

int main(void) {
    printf("Classic CPU code...\n");
    
	helloFromGPU <<<1, 1>>>();
	float *a, *b, *c;
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	c = (float*)malloc(sizeof(float) * N);
	gpu_matrix_mult <<<1, 8>>>(a,b,c,8,8,8);
	cudaDeviceReset();

    return 0;
}
