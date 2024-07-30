#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <assert.h>
#include <math.h>
#include <functional>
#include <numeric>
#include <vector>

#define NUM_RUNS 100

template<typename Func, typename... Args>
float time_function(int N, Func func, Args... args) {
    std::vector<float> durations(N);

    for (int i = 0; i < N; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        
        func(args...);  // Call the function
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        durations[i] = milliseconds;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Calculate average
    float sum = 0.0f;
    for (float d : durations) {
        sum += d;
    }
    float average = sum / N;


    std::cout << "Function ran " << N << " times.\n";
    std::cout << "Average time: " << average << " ms\n";

    return average;
}
__global__ void double_matrix_x_only(const float *a, float *res, int sz) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < sz) {
        res[tid] = a[tid] * 2;
    }
}

__global__ void double_matrix_x_y(const float *a, float *res, int rows, int cols) {
    int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx;
    if (tid_x < cols && tid_y < rows) {
        idx = tid_y * cols + tid_x;

        res[idx] = a[idx] * 2;
    }
}

void serial_func(const float *a, float *res, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j  = 0; j < cols; j++) {
            res[j * cols + i] = a[j * cols + i] * 2;
        }
    }
}

void create_a(float *a, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j  = 0; j < cols; j++) {
            a[j * cols + i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

int main() {

    int rows = 2500;
    int cols = 2500;

    long long int totalBytes = rows * cols * sizeof(float);

    float *h_a = (float*)malloc(totalBytes);
    float *h_res_x_only = (float*)malloc(totalBytes);
    float *h_res_x_y = (float*)malloc(totalBytes);
    float *h_res_serial = (float*)malloc(totalBytes);

    float *d_a, *d_res_x_only, *d_res_x_y;

    cudaMalloc(&d_a, totalBytes);
    cudaMalloc(&d_res_x_only, totalBytes);
    cudaMalloc(&d_res_x_y, totalBytes);

    create_a(h_a, rows, cols);

    cudaMemcpy(d_a, h_a, totalBytes, cudaMemcpyHostToDevice);

    time_function(NUM_RUNS, serial_func, h_a, h_res_serial, rows, cols);

    cudaDeviceSynchronize();

    auto x_only_cuda_wrapper = [&] (int blockSize) {
        double_matrix_x_only<<<ceil((rows*cols)/blockSize), blockSize>>>(d_a, d_res_x_only, rows*cols);
    };

    time_function(NUM_RUNS, x_only_cuda_wrapper, 256);
    cudaMemcpy(h_res_x_only, (void *) d_res_x_only, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_res_x_only);

    cudaDeviceSynchronize();

    auto x_y_cuda_wrapper = [&] (int blockSize) {
        dim3 threadsPerBlock(blockSize, blockSize, 1);
        dim3 numBlocks(ceil(cols/threadsPerBlock.x), ceil(rows/threadsPerBlock.y), 1);

        double_matrix_x_y<<<numBlocks, threadsPerBlock>>>(d_a, d_res_x_y, rows, cols);
    };

    time_function(NUM_RUNS, x_y_cuda_wrapper, 16);

    cudaMemcpy(h_res_x_y, (void *) d_res_x_y, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_res_x_y);

    cudaFree(d_a);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = j * cols + i;
            assert(h_res_serial[idx] == h_a[idx] * 2);
        }
    }

    free(h_res_serial);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = j * cols + i;
            assert(h_res_x_only[idx] == h_a[idx] * 2);
        }
    }

    free(h_res_x_only);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = j * cols + i;
            assert(h_res_x_y[idx] == h_a[idx] * 2);
        }
    }

    free(h_res_x_y);
    free(h_a);

    return 0;

}