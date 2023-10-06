// STDLIB
#include <cassert>
#include <iostream>
#include <vector>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// LOCAL
#include "eigen_blas.h"
#include "blas.cu.h"
#include "util.h"

int main()
{
    using namespace Util;

    int rows, cols;
    rows = 64;
    cols = 32;

    auto x = random_vec (rows);
    auto y = random_vec (rows);
    float alpha = 2.0;

    float *x_d = copy_to_cuda (x);
    float *y_d = copy_to_cuda (y);

    EigenImpl::axpy (rows, alpha, x.data (), y.data ());
    CudaImpl::axpy (rows, alpha, x_d, y_d);

    auto y_h = copy_from_cuda (y_d, rows);
    cudaFree (x_d);
    cudaFree (y_d);

    assert (("axpy test", compare (y, y_h)));

    auto A = random_vec (rows * cols);
    y = random_vec (rows);
    x = random_vec (cols);
    float beta = 0.5;

    float* A_d = copy_to_cuda (A);
    x_d = copy_to_cuda (x);
    y_d = copy_to_cuda (y);

    EigenImpl::gemv (rows, cols, alpha, A.data (), x.data (), beta, y.data ());
    CudaImpl::gemv (rows, cols, alpha, A_d, x_d, beta, y_d);

    y_h = copy_from_cuda (y_d, rows);
    cudaFree (x_d);
    cudaFree (y_d);
    cudaFree (A_d);

    assert (("gemv test", compare (y, y_h)));

    A = random_vec (rows * cols);
    auto B = random_vec (cols * cols);
    auto C = random_vec (rows * cols);

    A_d = copy_to_cuda (A);
    float* B_d = copy_to_cuda (B);
    float* C_d = copy_to_cuda (C);

    EigenImpl::gemm (rows, cols, cols, alpha, A.data (), B.data (), beta, C.data ());
    CudaImpl::gemm (rows, cols, cols, alpha, A_d, B_d, beta, C_d);

    auto C_h = copy_from_cuda (C_d, rows * cols);
    cudaFree (A_d);
    cudaFree (B_d);
    cudaFree (C_d);

    assert(("gemm test", compare (C, C_h)));

    std::cout << "Tests passed\n";
}
