#ifndef UTIL_H_
#define UTIL_H_

#include <vector>

// -----------------
// Utility functions
// -----------------

namespace Util {

/**
 * Return vector with elements = 0..size-1
 */
std::vector<float> range (size_t size);

/**
 * Return vector of given size with a constant value
 */
std::vector<float> init_host (size_t size, float value);

/**
 * Return vector of given size with random uniform values between 0 and 1
 */
std::vector<float> random_vec (size_t size);

/**
 * Return vector of size width*width with diagonal elements equal to 1.0
 */
std::vector<float> ident (size_t width);

/**
 * Return l2 norm of vector
 */
float norm (const std::vector<float>& v);

/**
 * Compare two vectors, return true if the same within bounds
 */
bool compare (const std::vector<float>& v1,
              const std::vector<float>& v2);

/**
 * Copy data to newly allocated device memory and return device addr
 */
float* copy_to_cuda (const std::vector<float>& v);

/**
 * Copy data to host from device and return data. Does not free device memory.
 */
std::vector<float> copy_from_cuda (float* v_dev, size_t size);

} // namespace


#endif // UTIL_H_
