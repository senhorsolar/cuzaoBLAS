#ifndef EIGEN_BLAS_H_
#define EIGEN_BLAS_H_

#include <cstdint>

namespace EigenImpl {

void axpy (int64_t n,
           float alpha,
           float* x,
           float* y);

void gemv (int64_t m,
           int64_t n,
           float alpha,
           float* A,
           float* x,
           float beta,
           float* y);

void gemm (int64_t m,
           int64_t n,
           int64_t k,
           float alpha,
           float* A,
           float* B,
           float beta,
           float* C);

} // namespace

#endif // EIGEN_BLAS_H_
