#include <iostream>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    cudaDeviceProp prop;

     int count;
     cudaGetDeviceCount(&count);

     printf("Device count: %d\n", count);

     for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execution timeout: ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Total global memory: %ld\n", prop.totalGlobalMem);
        printf("Total constant memory: %ld\n", prop.totalConstMem);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared memory per block: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max block dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
     }
}