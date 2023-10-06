// SYSTEM
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// LOCAL
#include "util.h"

namespace Util {

std::vector<float> range (size_t size)
{
    std::vector<float> v (size);
    std::iota (v.begin (), v.end (), 0.0);
    return v;
}

std::vector<float> init_host (size_t size, float value)
{
    std::vector<float> v (size, value);
    return v;
}

std::vector<float> random_vec (size_t size)
{
    std::default_random_engine g_gen;
    std::uniform_real_distribution<float> g_dist(0.0, 1.0);
    std::vector<float> v (size);
    for (auto&x : v) {
        x = g_dist (g_gen);
    }
    return v;
}

std::vector<float> ident (size_t width)
{
    std::vector<float> v (width * width, 0.0);
    for (size_t i=0; i < width; ++i) {
        v.at (i * width + i) = 1.0;
    }
    return v;
}

float norm (const std::vector<float>& v)
{
    float norm_val = std::sqrt (
        std::inner_product (v.begin (), v.end (),
                            v.begin (), 0.0));
    return norm_val;
}

bool compare (const std::vector<float>& v1,
              const std::vector<float>& v2)
{
    double eps = 1e-6;
    if (v1.size () != v2.size ()) {
        std::cerr << "v1 and v2 sizes not the same\n";
        return false;
    }
    if (std::abs (norm (v1) - norm (v2)) < eps) {
        return true;
    }
    else {
        bool ret = true;
        for (size_t i = 0; i < v1.size (); ++i) {
            if (std::abs (v1[i] - v2[i]) >= eps) {
                std::cerr << "diff at idx " << i << ". ";
                std::cerr << "v1: " << v1[i] << ",";
                std::cerr << "v2: " << v2[i] << ".\n";
                ret = false;
            }
        }
        return ret;
    }
}

float* copy_to_cuda (const std::vector<float>& v)
{
    float *v_dev;
    ssize_t size = v.size () * sizeof (float);
    cudaMalloc ((void **) &v_dev, size);
    cudaMemcpy (v_dev, v.data (), size, cudaMemcpyHostToDevice);
    return v_dev;
}

std::vector<float> copy_from_cuda (float* v_dev, size_t size)
{
    std::vector<float> v (size);
    cudaMemcpy (v.data (), v_dev, size * sizeof (float), cudaMemcpyDeviceToHost);
    return v;
}

} // namespace
