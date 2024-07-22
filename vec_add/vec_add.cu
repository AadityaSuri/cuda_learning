#include <iostream>
#include <math.h>
#include <chrono>

__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vec_add_cuda(float *a, float *b, float *c, int n) {

    int size = n * sizeof(float);
    float *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    add<<<ceil(n / 256.0), 256>>>(dev_a, dev_b, dev_c, n);

    cudaMemcpy(dev_c, c, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void vec_add_normal(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    const int n = 100000;
    float a[n], b[n], c_cuda[n], c_normal[n];

    for (int i = 0; i < n; i++) {
        a[i] = (float)rand()/(float)(RAND_MAX/10000.0);
        b[i] = (float)rand()/(float)(RAND_MAX/10000.0);
        c[i] = 0;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    vec_add_cuda(a, b, c_cuda, n);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Time taken by CUDA add: " << duration << " microseconds" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    vec_add_normal(a, b, c_normal, n);
    t2 = std::chrono::high_resolution_clock::now();

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Time taken by normal add: " << duration2 << " microseconds" << std::endl;

    for (int i = 0; i < n; i++) {
        assert(c_cuda[i] == c_normal[i]);
    }

}



    
