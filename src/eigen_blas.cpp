#include <Eigen/Dense>
#include <iostream>
#include "eigen_blas.h"

namespace EigenImpl {

typedef Eigen::Matrix<float,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> MatrixXfRowMajor;

void axpy (int64_t n,
           float alpha,
           float* x,
           float* y)
{
    Eigen::Map<Eigen::VectorXf> x_ (x, Eigen::Index (n));
    Eigen::Map<Eigen::VectorXf> y_ (y, Eigen::Index (n));
    y_.array () += alpha * x_.array ();
}

void gemv (int64_t m,
           int64_t n,
           float alpha,
           float* A,
           float* x,
           float beta,
           float* y)
{
    Eigen::Map<Eigen::VectorXf> x_ (x, Eigen::Index (n));
    Eigen::Map<Eigen::VectorXf> y_ (y, Eigen::Index (m));
    Eigen::Map<MatrixXfRowMajor> A_ (A, Eigen::Index (m), Eigen::Index (n));

    y_.array () *= beta;
    y_.array () += (A_ * x_).array () * alpha;
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
    Eigen::Map<MatrixXfRowMajor> A_ (A, Eigen::Index (m), Eigen::Index (k));
    Eigen::Map<MatrixXfRowMajor> B_ (B, Eigen::Index (k), Eigen::Index (n));
    Eigen::Map<MatrixXfRowMajor> C_ (C, Eigen::Index (m), Eigen::Index (n));

    C_.array () *= beta;
    C_.array () += (A_ * B_).array () * alpha;
}

} // namespace
