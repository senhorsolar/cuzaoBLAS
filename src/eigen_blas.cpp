#include "eigen_blas.h"

namespace EigenImpl {

void axpy (int64_t n,
           float alpha,
           float* x,
           float* y)
{
    Eigen::VectorXd x_ = Eigen::Map<Eigen::VectorXd> (x, n);
    Eigen::VectorXd y_ = Eigen::Map<Eigen::VectorXd> (y, n);
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
    Eigen::VectorXd x_ = Eigen::Map<Eigen::VectorXd> (x, n);
    Eigen::VectorXd y_ = Eigen::Map<Eigen::VectorXd> (y, m);
    Eigen::MatrixXd A_ = Eigen::Map<Eigen::MatrixXd> (A, m, n);

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
    Eigen::MatrixXd A_ = Eigen::Map<Eigen::MatrixXd> (A, m, k);
    Eigen::MatrixXd B_ = Eigen::Map<Eigen::MatrixXd> (B, k, n);
    Eigen::MatrixXd C_ = Eigen::Map<Eigen::MatrixXd> (C, m, n);

    C_.array () *= beta;
    C_.array () += (A_ * B_).array () * alpha;
}

} // namespace
