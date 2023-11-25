#include <stdio.h>

__global__ void helloFromGPU (void) {
    printf("and hello from A30 GPU!\n");
}

int main(void) {
    printf("Classic CPU code...\n");
    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
