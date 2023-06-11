#include <iostream>
#include <chrono>
#define N 250000

void __global__ add_cuda(long int *a, long int *b, long int *c) {
    long int tid = blockIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

void add_normal(long int *a, long int *b, long int *c) {
    for (long int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    long int a[N], b[N], c_cuda[N], c_normal[N];
    long int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, N * sizeof(long int));
    cudaMalloc((void**)&dev_b, N * sizeof(long int));
    cudaMalloc((void**)&dev_c, N * sizeof(long int));

    for (long int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(long int), cudaMemcpyHostToDevice);

    auto t1 = std::chrono::high_resolution_clock::now();
    add_cuda<<<N, 1>>>(dev_a, dev_b, dev_c);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Time taken by CUDA: " << duration << " microseconds" << std::endl;

    cudaMemcpy(c_cuda, dev_c, N * sizeof(long int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    // }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    auto t3 = std::chrono::high_resolution_clock::now();
    add_normal(a, b, c_normal);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    std::cout << "Time taken by normal: " << duration2 << " microseconds" << std::endl;


    return 0;
}