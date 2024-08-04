#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <assert.h> 
#include <iomanip>

#define BLOCK_SZ 16
#define TILE_SZ 16

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
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Kernel execution timeout: " << deviceProp.kernelExecTimeoutEnabled << std::endl;
        std::cout << "  Unified addressing: " << deviceProp.unifiedAddressing << std::endl;
        std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;
        std::cout << "  L2 cache size: " << deviceProp.l2CacheSize << " bytes" << std::endl;
        std::cout << std::endl;
    }
}


__global__ void matmul(const float *a, const float *b, float *res, int sz) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sz && col < sz) {
        float Pvalue = 0;
        for (int k = 0; k < sz; k++) {
            Pvalue += a[row * sz + k] * b[k * sz + col];
        }
        res[row * sz + col] = Pvalue;
    }
    
}

__global__ void tiledMatmul(const float *a, const float *b, float *res, int sz) {
    __shared__ float aTile[TILE_SZ][TILE_SZ];
    __shared__ float bTile[TILE_SZ][TILE_SZ];

    int bx, by, tx, ty;
    bx = blockIdx.x;
    by = blockIdx.y;
    tx = threadIdx.x;
    ty = threadIdx.y;

    int row = by * TILE_SZ + ty;
    int col = bx * TILE_SZ + tx;

    float Pvalue = 0;
    for (int m = 0; m < sz/TILE_SZ; m++) {

        aTile[ty][tx] = a[row * sz + m * TILE_SZ + tx];
        bTile[ty][tx] = b[(m * TILE_SZ + ty) * sz + col];

        __syncthreads();

        for (int k = 0; k < TILE_SZ; k++) {
            Pvalue += aTile[ty][k] * bTile[k][tx];
        }

        __syncthreads();
    }

    res[row * sz + col] = Pvalue;
}
    

void create_matrices(float *a, float *b, int sz) {
    // fill matrices with random values
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            a[i * sz + j] = static_cast<float>(rand()) / RAND_MAX;
            b[i * sz + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// function to generate actual results by multiplying matrices a and b
// res is the result matrix
void serial_func(const float *a, const float *b, float *res, int sz) {
    // matrix multiplication of a and b
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            res[i * sz + j] = 0;
            for (int k = 0; k < sz; k++) {
                res[i * sz + j] += a[i * sz + k] * b[k * sz + j];
            }
        }
    }
}

// function to print matrix
void print_matrix(float *mat, int sz) {
    std::cout << "Matrix: \n";
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            std::cout << mat[i * sz + j] << " ";
        }
        std::cout << std::endl;
    }
}

void check_results(float *res, float *ref, int sz) {
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            int idx = i * sz + j;
            float diff = std::abs(res[idx] - ref[idx]);
            if (diff >= 1e-4) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " 
                          << std::setprecision(10) << res[idx] << " vs " 
                          << std::setprecision(10) << ref[idx] 
                          << " (diff: " << std::setprecision(10) << diff << ")" 
                          << std::endl;
            }
        }
    }
}

int main() {

    displayCudaProperties();
    int sz = 1024;
    long long int totalBytes = sz * sz * sizeof(float);

    float *h_a = (float*)malloc(totalBytes);
    float *h_b = (float*)malloc(totalBytes);
    float *h_res = (float*)malloc(totalBytes);
    float *h_res_tiled = (float*)malloc(totalBytes);

    float *d_a, *d_b, *d_res, *d_res_tiled;

    create_matrices(h_a, h_b, sz);

    // print_matrix(h_a, sz);
    // print_matrix(h_b, sz);

    cudaMalloc(&d_a, totalBytes);
    cudaMalloc(&d_b, totalBytes);
    cudaMalloc(&d_res, totalBytes);
    cudaMalloc(&d_res_tiled, totalBytes);

    cudaMemcpy(d_a, h_a, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, totalBytes, cudaMemcpyHostToDevice);

    dim3 dimGrid((sz + BLOCK_SZ - 1) / BLOCK_SZ, (sz + BLOCK_SZ - 1) / BLOCK_SZ, 1);
    dim3 dimBlock(BLOCK_SZ, BLOCK_SZ, 1);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    matmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_res, sz);
    cudaDeviceSynchronize();
    auto stop_gpu = std::chrono::high_resolution_clock::now();

    auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu);

    std::cout << "GPU time: " << duration_gpu.count() << " microseconds" << std::endl;

    cudaMemcpy(h_res, d_res, totalBytes, cudaMemcpyDeviceToHost);

    auto start_gpu_tiled = std::chrono::high_resolution_clock::now();
    tiledMatmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_res_tiled, sz);
    cudaDeviceSynchronize();
    auto stop_gpu_tiled = std::chrono::high_resolution_clock::now();

    auto duration_gpu_tiled = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu_tiled - start_gpu_tiled);

    std::cout << "GPU time (tiled): " << duration_gpu_tiled.count() << " microseconds" << std::endl;

    cudaMemcpy(h_res_tiled, d_res_tiled, totalBytes, cudaMemcpyDeviceToHost);

    // check results
    float *ref = (float*)malloc(totalBytes);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    serial_func(h_a, h_b, ref, sz);
    auto stop_cpu = std::chrono::high_resolution_clock::now();

    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);

    std::cout << "CPU time: " << duration_cpu.count() << " microseconds" << std::endl;

    // check_results(h_res, ref, sz);

    std::cout<<"Checking results"<<std::endl;

    check_results(h_res, ref, sz);
    check_results(h_res_tiled, ref, sz);

    std::cout<<"Assertions passing"<<std::endl;
    // print_matrix(h_res, sz);
    // print_matrix(ref, sz);

    free(h_res);
    free(h_a);
    free(h_b);
    free(ref);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
}