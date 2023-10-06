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

    // TILING: Peform dot product in phases
    for (int64_t ph = 0; ph < ceil(n/(float)FLAT_WIDTH); ++ph) {
        // Boundary check on x
        if (row < m && (ph * FLAT_WIDTH + tx) < n) {
            xds[tx] = x[ph * FLAT_WIDTH + tx];
        }
        else if (tx < FLAT_WIDTH) {
            xds[tx] = 0.0f;
        }

        __syncthreads();

        if (row < m) {
            // Perform dot product for current phase
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

    // TILING: Peform dot product in phases
    for (int ph = 0; ph < ceil(k/(float)TILE_WIDTH); ++ph) {
        // Boundary check on A
        if (row < m && (ph*TILE_WIDTH+tx) < k) {
            Ads[ty][tx] = A[row * k + ph*TILE_WIDTH + tx];
        }
        else if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
            Ads[ty][tx] = 0.0f;
        }
        // Boundary check on B
        if ((ph*TILE_WIDTH+ty) < k && col < n) {
            Bds[ty][tx] = B[(ph*TILE_WIDTH + ty)*n + col];
        }
        else if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
            Bds[ty][tx] = 0.0f;
        }
        // ensure data loaded into shared memory for all threads
        __syncthreads();

        for (int k_ = 0; k_ < TILE_WIDTH; ++k_) {
            p += alpha * Ads[ty][k_] * Bds[k_][tx];
        }
        // ensure shared memory used before overwritten
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = p + beta * C[row * n + col];
    }
}

} // namespace Kernel

namespace CudaImpl {

void axpy (int64_t n,
           float alpha,
           float* x,
           float* y)
{
    Kernel::axpy<<<ceil (n/(float)FLAT_WIDTH), FLAT_WIDTH>>> (
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
    Kernel::gemv<<<ceil (n/(float)FLAT_WIDTH), FLAT_WIDTH>>> (
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
