#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CHECK_CUDA(call)                                      \
{                                                              \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess)                                    \
    {                                                          \
        std::cerr << "CUDA Error: "                            \
                  << cudaGetErrorString(err)                   \
                  << " at line " << __LINE__ << std::endl;     \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

#define CHECK_CUSOLVER(call)                                   \
{                                                              \
    cusolverStatus_t status = call;                            \
    if (status != CUSOLVER_STATUS_SUCCESS)                     \
    {                                                          \
        std::cerr << "cuSOLVER Error at line "                 \
                  << __LINE__ << std::endl;                    \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

int main()
{
    // Solve:
    //
    // [3 1] [x1] = [9]
    // [1 2] [x2]   [8]
    //
    // Expected:
    // x1 = 2
    // x2 = 3

    const int N = 2;
    const int lda = N;
    const int ldb = N;

    // Column-major format for cuSOLVER
    std::vector<float> h_A =
    {
        3, 1,
        1, 2
    };

    std::vector<float> h_b =
    {
        9,
        8
    };

    float* d_A;
    float* d_b;

    CHECK_CUDA(cudaMalloc((void**)&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
        d_A,
        h_A.data(),
        N * N * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_b,
        h_b.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    cusolverDnHandle_t solverHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&solverHandle));

    int work_size = 0;

    CHECK_CUSOLVER(
        cusolverDnSgetrf_bufferSize(
            solverHandle,
            N,
            N,
            d_A,
            lda,
            &work_size));

    float* d_work;
    CHECK_CUDA(cudaMalloc((void**)&d_work,
                          work_size * sizeof(float)));

    int* d_ipiv;
    int* d_info;

    CHECK_CUDA(cudaMalloc((void**)&d_ipiv,
                          N * sizeof(int)));

    CHECK_CUDA(cudaMalloc((void**)&d_info,
                          sizeof(int)));

    // LU decomposition
    CHECK_CUSOLVER(
        cusolverDnSgetrf(
            solverHandle,
            N,
            N,
            d_A,
            lda,
            d_work,
            d_ipiv,
            d_info));

    // Solve Ax = b
    CHECK_CUSOLVER(
        cusolverDnSgetrs(
            solverHandle,
            CUBLAS_OP_N,
            N,
            1,
            d_A,
            lda,
            d_ipiv,
            d_b,
            ldb,
            d_info));

    // Copy solution back
    std::vector<float> h_x(N);

    CHECK_CUDA(cudaMemcpy(
        h_x.data(),
        d_b,
        N * sizeof(float),
        cudaMemcpyDeviceToHost));

    std::cout << "\nSolution:\n";
    std::cout << "x1 = " << h_x[0] << std::endl;
    std::cout << "x2 = " << h_x[1] << std::endl;

    // Cleanup
    cusolverDnDestroy(solverHandle);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_work);
    cudaFree(d_ipiv);
    cudaFree(d_info);

    return 0;
}