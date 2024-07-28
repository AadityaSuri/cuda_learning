#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

void displayCudaProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: (" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate << " kHz" << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem << " bytes" << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << std::endl;
    }
}

// CUDA kernel for vector addition
__global__ void vectorAddKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function for vector addition
void vectorAddHost(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Utility function to initialize vectors
void initializeVectors(float *a, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to check results
void checkResults(const float *c1, const float *c2, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(c1[i] - c2[i]) > 1e-5) {
            std::cerr << "Results do not match!" << std::endl;
            std::cout << c1[i] << " " << c2[i] << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}

int main() {
    // displayCudaProperties();


    int n = 1 << 24; // Vector size
    size_t bytes = n * sizeof(float);

    // Allocate memory on host
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_c_gpu = (float *)malloc(bytes);

    // Initialize vectors
    initializeVectors(h_a, h_b, n);

    // Allocate memory on device
    float *d_a, *d_b, *d_c;


    int num_runs = 100;

    // Launch CUDA kernel
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;

    // dim3 dimBlock(128, 1, 1);
    // dim3 dimGrid(32, 1, 1);    


    auto start_gpu = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < num_runs; i++) {
        vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    }

    cudaDeviceSynchronize();
    auto stop_gpu = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    // Time host function
    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0 ; i < num_runs; i++) {
        vectorAddHost(h_a, h_b, h_c, n);
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++) {
        std::cout << h_c[i] << " " << h_c_gpu[i] << std::endl;
    }

    // Check results
    checkResults(h_c, h_c_gpu, n);

    // Calculate elapsed times
    auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu) / num_runs;
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu) / num_runs;

    std::cout << "GPU time: " << duration_gpu.count() << " microseconds" << std::endl;
    std::cout << "CPU time: " << duration_cpu.count() << " microseconds" << std::endl;

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}
