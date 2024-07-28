#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

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

int main() {

    int rows = 50000;
    int cols = 50000;

    long long int totalBytes = rows * cols * sizeof(float);

    float *h_a = (float*)malloc(totalBytes);
    float *h_res_x_only = (float*)malloc(totalBytes);
    float *h_res_x_y = (float*)malloc(totalBytes);
    float *h_res_serial = (float*)malloc(totalBytes);

    float *d_a, *d_res_x_only, *d_res_x_y, *d_res_serial;

    return 0;

}