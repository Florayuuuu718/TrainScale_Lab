// Minimal CUDA C++ toolchain smoke test for module 02.
//
// This program deliberately avoids PyTorch headers.  Its only job is to prove
// that nvcc can generate code for the current GPU and that the driver can load
// and execute that code.  The later operator benchmark adds framework bindings.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expression)                                                   \
  do {                                                                           \
    const cudaError_t error = (expression);                                      \
    if (error != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "      \
                << cudaGetErrorString(error) << '\n';                            \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (false)

__global__ void vector_add(const float* left, const float* right, float* output,
                           int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    output[index] = left[index] + right[index];
  }
}

int main() {
  constexpr int size = 4097;  // A prime-ish ragged size exercises the tail guard.
  constexpr int threads = 256;
  const std::size_t bytes = size * sizeof(float);

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::vector<float> left(size);
  std::vector<float> right(size);
  std::vector<float> actual(size);
  for (int index = 0; index < size; ++index) {
    left[index] = static_cast<float>(index) * 0.25F;
    right[index] = static_cast<float>(index % 17) - 8.0F;
  }

  float* device_left = nullptr;
  float* device_right = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_left, bytes));
  CUDA_CHECK(cudaMalloc(&device_right, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = (size + threads - 1) / threads;
  vector_add<<<blocks, threads>>>(device_left, device_right, device_output, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  float max_absolute_error = 0.0F;
  for (int index = 0; index < size; ++index) {
    float error = actual[index] - (left[index] + right[index]);
    error = error < 0.0F ? -error : error;
    max_absolute_error = std::max(max_absolute_error, error);
  }

  CUDA_CHECK(cudaFree(device_left));
  CUDA_CHECK(cudaFree(device_right));
  CUDA_CHECK(cudaFree(device_output));

  if (max_absolute_error != 0.0F) {
    std::cerr << "CUDA C++ smoke failed: max absolute error=" << max_absolute_error << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "CUDA C++ smoke passed: " << size << " elements on " << properties.name
            << " (SM " << properties.major << '.' << properties.minor << ")\n";
  return EXIT_SUCCESS;
}
