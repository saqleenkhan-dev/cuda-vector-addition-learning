
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    vectorAdd<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    bool correct = true;

    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "PASS" : "FAIL") << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
