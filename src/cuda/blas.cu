/*
 * blas.cu
 */

#include "blas.cu.h"
#include <__clang_cuda_builtin_vars.h>
#include <cstdint>

#define TILE_WIDTH 16
#define FLAT_WIDTH 256

namespace CudaImpl {

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

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = blockIdx.x;
    int ty = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float p = 0;

    for (int ph = 0; ph < ceil(k/(float)TILE_WIDTH); ++ph) {
        if (row < m && (ph*TILE_WIDTH+tx) < k) {
            Ads[ty][tx] = A[row * k + ph*TILE_WIDTH + tx];            
        }
        else {
            Ads[ty][tx] = 0.0f;
        }
        if ((ph*TILE_WIDTH+ty) < k && col < n) {
            Bds[ty][tx] = B[(ph*TILE_WIDTH + ty)*n + col];
        }
        else {
            Bds[ty][tx] = 0.0f;            
        }
        __syncthreads();

        for (int k_ = 0; k_ < TILE_WIDTH; ++k_) {
            p += alpha * Ads[ty][k_] * Bds[k_][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * m + col] = p + beta * C[row * m + col];
    }
}

} // namespace
