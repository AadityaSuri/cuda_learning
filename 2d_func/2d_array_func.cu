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
float time_function(std::string function_name, int N, Func func, Args... args) {
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


    std::cout << "Function " << function_name << ": \n";
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
            res[i * cols + j] = a[i * cols + j] * 2;
        }
    }
}

void create_a(float *a, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void check_results(float *res, float *ref, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            assert(res[idx] == ref[idx] * 2);
        }
    }
}

int main() {

    int rows = 10000;
    int cols = 10000;

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

    time_function("no_cuda", NUM_RUNS, serial_func, h_a, h_res_serial, rows, cols);

    cudaDeviceSynchronize();

    auto x_only_cuda_wrapper = [&] (int blockSize) {
        int dimSize = (rows * cols + blockSize - 1) / blockSize;
        double_matrix_x_only<<<dimSize, blockSize>>>(d_a, d_res_x_only, rows*cols);
    };

    time_function("1d_cuda",NUM_RUNS, x_only_cuda_wrapper, 256);
    cudaMemcpy(h_res_x_only, d_res_x_only, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_res_x_only);

    cudaDeviceSynchronize();

    auto x_y_cuda_wrapper = [&] (int blockSize) {
        dim3 threadsPerBlock(blockSize, blockSize, 1);
        dim3 numBlocks(
            (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
             1);

        double_matrix_x_y<<<numBlocks, threadsPerBlock>>>(d_a, d_res_x_y, rows, cols);
    };

    time_function("2d_cuda", NUM_RUNS, x_y_cuda_wrapper, 16);

    cudaMemcpy(h_res_x_y, (void *) d_res_x_y, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_res_x_y);

    cudaFree(d_a);

    std::cout<<"Checking results"<<std::endl;

    check_results(h_res_serial, h_a, rows, cols);
    check_results(h_res_x_only, h_a, rows, cols);
    check_results(h_res_x_y, h_a, rows, cols);

    std::cout<<"Assertions passing"<<std::endl;

    free(h_res_x_y);
    free(h_res_serial);
    free(h_res_x_only);
    free(h_a);

    return 0;

}