// All programs here require the Nvidia GPU and cuda-toolkit from:
// https://developer.nvidia.com/cuda-downloads?target_os=Linux
// Compile this one using command:   nvcc agpu1.cu -o agpu1 

#include <stdio.h>
#include <iostream>
using namespace std;

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    wcout << "==========" << endl << " The.GPUs" << endl << "==========" << endl << endl;

    wcout << "CUDA version:   v" << CUDART_VERSION << endl;    
//    wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    wcout << "CUDA Devices: " << endl << endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        wcout << "  Global memory:   " << props.totalGlobalMem / mb << " MB" << endl;
        wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << " KB" << endl;
        wcout << "  Constant memory: " << props.totalConstMem / kb << " KB" << endl;
        wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

        wcout << "  Warp size:         " << props.warpSize << endl;
        wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        wcout << endl;
    }
}


int main(void) {
    printf("detecting devices...\n");
    DisplayHeader();
    cudaDeviceReset();
    return 0;
}
