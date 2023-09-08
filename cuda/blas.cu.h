/*
 * blas.cu.h
 */

#ifndef BLAS_CU_H_
#define BLAS_CU_H_

/**
 * Basic level 1 routine
 *  y <- a*x + y, a generalized vector addition
 *
 * @param n: Number of elements in vector x
 * @param alpha: Scaling factor for vector x
 * @param x: Input vector x
 * @param y: Output vector y
 */
__global__
void axpy (int64_t n, 
           float alpha, 
           float* x, 
           float* y);

/**
 * Basic level 2 routine
 *  y <- a*A*x + b*y, generalized matrix-vector multiplication
 *
 * @param m: Number of rows of matrix A
 * @param n: Number of columns of matrix A
 * @param alpha: Scaling factor for matrix-vector product
 * @param A: Input matrix A in matrix-vector product
 * @param x: Input vector x in matrix-vector product
 * @param beta: Scaling factor for vector y
 * @param y: Output vector y
 */
__global__
void gemv (int64_t m, 
           int64_t n, 
           float alpha, 
           float* A, 
           float* x, 
           float beta, 
           float* y);

/**
 * Basic level 3 routine
 *  C <- a*A*B + b*C, a general matrix multiplication
 *
 *  @param m: Number of rows of matrix A
 *  @param n: Number of columns of matrix B
 *  @param k: Number of rows of matrix B
 *  @param alpha: Scaling factor for matrix-matrix product
 *  @param A: Input matrix A in matrix-matrix product
 *  @param B: Input matrix B in matrix-matrix product
 *  @param beta: Scaling factor for matrix C
 *  @param C: Output matrix C
 */
__global__
void gemm (int64_t m,
           int64_t n,
           int64_t k,
           float alpha,
           float* A,
           float* B,
           float beta,
           float* C);

#endif /* BLAS_CU_H_ */
