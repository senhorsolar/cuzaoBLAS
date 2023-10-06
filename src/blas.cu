/*
 * blas.cu
 */

#include "blas.cu.h"
#include <cstdint>
#include <cstdio>

#define TILE_WIDTH 16
#define FLAT_WIDTH 256

namespace Kernel {

__global__
void axpy (int64_t n,
           float alpha,
           float* x,
           float* y)
{
    int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

__global__
void gemv (int64_t m,
           int64_t n,
           float alpha,
           float* A,
           float* x,
           float beta,
           float* y)
{
    __shared__ float xds[FLAT_WIDTH];

    int64_t row = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t tx = threadIdx.x;

    float p = 0;

    for (int64_t ph = 0; ph < ceil(n/(float)FLAT_WIDTH); ++ph) {
        if (row < m && (ph * FLAT_WIDTH + tx) < n) {
            xds[tx] = x[ph * FLAT_WIDTH + tx];
        }
        else {
            xds[tx] = 0.0f;
        }

        __syncthreads();

        if (row < m) {
            for (int64_t i = 0, j = ph*FLAT_WIDTH; i < FLAT_WIDTH && j < n; ++i, ++j) {
                p += alpha * A[row * n + j] * xds[i];
            }
        }

        __syncthreads();
    }

    if (row < m) {
        y[row] = p + beta * y[row];
    }
}

__global__
void gemm (int64_t m,
           int64_t n,
           int64_t k,
           float alpha,
           float* A,
           float* B,
           float beta,
           float* C)
{
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = ty + blockDim.y * blockIdx.y;
    int col = tx + blockDim.x * blockIdx.x;

    float p = 0;

    for (int ph = 0; ph < ceil(k/(float)TILE_WIDTH); ++ph) {
        if (row < m && (ph*TILE_WIDTH+tx) < k) {
            Ads[ty][tx] = A[row * k + ph*TILE_WIDTH + tx];
            //printf("A is not zero\n");
        }
        else if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
            Ads[ty][tx] = 0.0f;
        }
        //printf("bx: %d, by: %d, tx: %d, ty: %d, (ph*TILE_WIDTH+ty): %d, k: %ld, col: %d, n: %ld\n",
        //       bx, by, tx, ty, (ph*TILE_WIDTH+ty), k, col, n);
        int b_row = ph * TILE_WIDTH + ty;
        if (b_row == col && b_row < k && col < n) {
            //printf("B(%d, %d)=%f\n", b_row, col, B[b_row*n + col]);
        }
        if ((ph*TILE_WIDTH+ty) < k && col < n) {
            Bds[ty][tx] = B[(ph*TILE_WIDTH + ty)*n + col];
            //printf("here: %f\n", Bds[ty][tx]);
        }
        else if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
            Bds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k_ = 0; k_ < TILE_WIDTH; ++k_) {
            p += alpha * Ads[ty][k_] * Bds[k_][tx];
        }
        if (p < 0 || p > 0) {
            //printf("here\n");
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = p + beta * C[row * n + col];
        //printf("C[%d,%d]=%f\n", row, col, C[row*m+col]);
    }
}

} // namespace Kernel

namespace CudaImpl {

void axpy (int64_t n,
           float alpha,
           float* x,
           float* y)
{
    Kernel::axpy<<<ceil (n/256.0), 256>>> (
        n,
        alpha,
        x,
        y);
}

void gemv (int64_t m,
           int64_t n,
           float alpha,
           float* A,
           float* x,
           float beta,
           float* y)
{
    Kernel::gemv<<<ceil (n/256.0), 256>>> (
        m,
        n,
        alpha,
        A,
        x,
        beta,
        y);
}

void gemm (int64_t m,
           int64_t n,
           int64_t k,
           float alpha,
           float* A,
           float* B,
           float beta,
           float* C)
{
    dim3 gd (ceil (n / (float)TILE_WIDTH), ceil (m / (float)TILE_WIDTH), 1);
    dim3 bd (TILE_WIDTH, TILE_WIDTH, 1);
    Kernel::gemm<<<gd, bd>>> (
        m,
        n,
        k,
        alpha,
        A,
        B,
        beta,
        C);
}

} // namespace
