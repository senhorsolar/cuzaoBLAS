// SYSTEM
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// LOCAL
#include "eigen_blas.h"
#include "blas.cu.h"
#include "util.h"

struct Times
{
    double eign_time;
    double cuda_time;
};

class Algo
{
public:

    virtual ~Algo ()=default;

    /**
     * Set size of algo (i.e., vector size)
     */
    virtual void SetSize (size_t n)=0;

    /**
     * Run one iteration of algo in eigen
     */
    virtual void RunAlgoEigen ()=0;

    /**
     * Run one iteration of algo in cuda
     */
    virtual void RunAlgoCuda ()=0;

    virtual void CopyToCuda ()=0;
    virtual void CopyFromCuda ()=0;

    /**
     * Name of algorithm
     */
    virtual std::string Name ()=0;

    /**
     * Time niters of algo runs
     */
    Times RunBothTimeTests (size_t niters)
    {
        double eign_time = RunTimeTest (niters, false);
        double cuda_time = RunTimeTest (niters, true);
        return {eign_time, cuda_time};
    }

    double RunTimeTest (size_t niters, bool cuda=false) {
        const auto start {std::chrono::steady_clock::now ()};
        if (cuda)
            CopyToCuda ();
        for (size_t i=0; i < niters; ++i) {
            if (cuda)
                RunAlgoCuda ();
            else
                RunAlgoEigen ();
        }
        if (cuda)
            CopyFromCuda ();
        const auto end {std::chrono::steady_clock::now ()};
        const std::chrono::duration<double> elapsed_seconds {end - start};
        return elapsed_seconds.count ();
    }
};

class Axpy : public Algo
{
public:

    Axpy () : m_alpha (2.0),
              m_x (),
              m_y (),
              m_xDevice (nullptr),
              m_yDevice (nullptr)
    {
    }

    ~Axpy () {
        cudaFree (m_xDevice);
        cudaFree (m_yDevice);
    }

    void SetSize (size_t n) override {
        m_x = Util::random_vec (n);
        m_y = Util::random_vec (n);

    }

    void CopyToCuda () override {
        if (m_xDevice != nullptr)
            cudaFree (m_xDevice);
        if (m_yDevice != nullptr)
            cudaFree (m_yDevice);
        m_xDevice = Util::copy_to_cuda (m_x);
        m_yDevice = Util::copy_to_cuda (m_y);
    }

    void CopyFromCuda () override {
        auto y_h = Util::copy_from_cuda (m_yDevice, m_y.size ());
        cudaFree (m_xDevice);
        cudaFree (m_yDevice);
    }

    void RunAlgoEigen () override {
        EigenImpl::axpy (m_x.size (), m_alpha, m_x.data (), m_y.data ());
    }

    void RunAlgoCuda () override {
        CudaImpl::axpy (m_x.size (), m_alpha, m_xDevice, m_yDevice);
    }

    std::string Name () override {
        return "axpy";
    }

private:
    double m_alpha;
    std::vector<float> m_x;
    std::vector<float> m_y;
    float* m_xDevice;
    float* m_yDevice;
};

class Gemv : public Algo
{
public:

    Gemv () : m_alpha (2.0),
              m_beta (0.5),
              m_x (),
              m_y (),
              m_A (),
              m_xDevice (nullptr),
              m_yDevice (nullptr),
              m_ADevice (nullptr)
    {
    }

    ~Gemv () {
        cudaFree (m_xDevice);
        cudaFree (m_yDevice);
        cudaFree (m_ADevice);
    }

    void SetSize (size_t n) override {
        m_x = Util::random_vec (n);
        m_y = Util::random_vec (n);
        m_A = Util::random_vec (n * n);
    }

    void CopyToCuda () override {
        if (m_xDevice != nullptr)
            cudaFree (m_xDevice);
        if (m_yDevice != nullptr)
            cudaFree (m_yDevice);
        if (m_ADevice != nullptr)
            cudaFree (m_ADevice);
        m_xDevice = Util::copy_to_cuda (m_x);
        m_yDevice = Util::copy_to_cuda (m_y);
        m_ADevice = Util::copy_to_cuda (m_A);
    }

    void CopyFromCuda () override {
        auto y_h = Util::copy_from_cuda (m_yDevice, m_y.size ());
        cudaFree (m_xDevice);
        cudaFree (m_yDevice);
        cudaFree (m_ADevice);
    }

    void RunAlgoEigen () override {
        EigenImpl::gemv (m_x.size (), m_x.size (), m_alpha,
                         m_A.data (), m_x.data (), m_beta, m_y.data ());
    }

    void RunAlgoCuda () override {
        CudaImpl::gemv (m_x.size (), m_x.size (), m_alpha,
                        m_ADevice, m_xDevice, m_beta, m_yDevice);
    }

    std::string Name () override {
        return "gemv";
    }

private:
    double m_alpha;
    double m_beta;
    std::vector<float> m_x;
    std::vector<float> m_y;
    std::vector<float> m_A;
    float* m_xDevice;
    float* m_yDevice;
    float* m_ADevice;
};

class Gemm : public Algo
{
public:

    Gemm () : m_alpha (2.0),
              m_beta (0.5),
              m_A (),
              m_B (),
              m_C (),
              m_ADevice (nullptr),
              m_BDevice (nullptr),
              m_CDevice (nullptr),
              m_size (0)
    {
    }

    ~Gemm () {
        cudaFree (m_ADevice);
        cudaFree (m_BDevice);
        cudaFree (m_CDevice);
    }

    void SetSize (size_t n) override {
        m_A = Util::random_vec (n * n);
        m_B = Util::random_vec (n * n);
        m_C = Util::random_vec (n * n);
        m_size = n;
    }

    void CopyToCuda () override {
        if (m_ADevice != nullptr)
            cudaFree (m_ADevice);
        if (m_BDevice != nullptr)
            cudaFree (m_BDevice);
        if (m_CDevice != nullptr)
            cudaFree (m_CDevice);
        m_ADevice = Util::copy_to_cuda (m_A);
        m_BDevice = Util::copy_to_cuda (m_B);
        m_CDevice = Util::copy_to_cuda (m_C);
    }

    void CopyFromCuda () override {
        auto C_h = Util::copy_from_cuda (m_CDevice, m_C.size ());
        cudaFree (m_ADevice);
        cudaFree (m_BDevice);
        cudaFree (m_CDevice);
    }

    void RunAlgoEigen () override {
        EigenImpl::gemm (m_size, m_size, m_size, m_alpha,
                         m_A.data (), m_B.data (), m_beta, m_C.data ());
    }

    void RunAlgoCuda () override {
        CudaImpl::gemm (m_size, m_size, m_size, m_alpha,
                        m_ADevice, m_BDevice, m_beta, m_CDevice);
    }

    std::string Name () override {
        return "gemm";
    }

private:
    double m_alpha;
    double m_beta;
    std::vector<float> m_A;
    std::vector<float> m_B;
    std::vector<float> m_C;
    float* m_ADevice;
    float* m_BDevice;
    float* m_CDevice;
    size_t m_size;
};

void LogInfo (const std::string& algo_name, size_t size, size_t niters, const Times& t)
{
    printf("%s\t%lu\t%lu\t%s\t%f\n", algo_name.c_str(), size, niters, "eign", t.eign_time);
    printf("%s\t%lu\t%lu\t%s\t%f\n", algo_name.c_str(), size, niters, "cuda", t.cuda_time);
}

int main()
{
    std::vector<size_t> sizes {1024, 2048, 4096};//, 8192, 16384, 32768};
    std::vector<size_t> niters {1};//, 10, 100, 500, 1000};
    std::vector<std::unique_ptr<Algo>> algos;
    algos.push_back (std::make_unique<Axpy> ());
    algos.push_back (std::make_unique<Gemv> ());
    algos.push_back (std::make_unique<Gemm> ());

    printf("algo\tsize\tniters\ttype\ttime\n");

    for (auto& algo : algos) {
        for (size_t size : sizes) {
            algo->SetSize (size);
            for (auto niter : niters) {
                Times t = algo->RunBothTimeTests (niter);
                LogInfo (algo->Name (), size, niter, t);
            }
        }
    }
}
