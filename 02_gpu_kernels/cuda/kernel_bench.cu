#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t error_ = (call);                                                       \
        if (error_ != cudaSuccess) {                                                       \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error_)); \
        }                                                                                 \
    } while (false)

struct Options {
    std::string operation;
    std::string variant = "baseline";
    std::string dtype = "float32";
    int size = 0;
    int rows = 0;
    int cols = 0;
    int samples = 21;
    int warmup = 10;
    int inner = 100;
};

double percentile(std::vector<float> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::llround(static_cast<double>(values.size() - 1) * fraction));
    return values[index];
}

template <typename T>
struct Numeric;

template <>
struct Numeric<float> {
    __host__ __device__ static float from_float(float value) { return value; }
    __host__ __device__ static float to_float(float value) { return value; }
};

template <>
struct Numeric<__half> {
    __host__ __device__ static __half from_float(float value) { return __float2half(value); }
    __host__ __device__ static float to_float(__half value) { return __half2float(value); }
};

template <>
struct Numeric<__nv_bfloat16> {
    __host__ __device__ static __nv_bfloat16 from_float(float value) {
        return __float2bfloat16(value);
    }
    __host__ __device__ static float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
};

template <typename T>
__global__ void vector_add_scalar(const T* x, const T* y, T* output, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        const float result = Numeric<T>::to_float(x[index]) + Numeric<T>::to_float(y[index]);
        output[index] = Numeric<T>::from_float(result);
    }
}

__global__ void vector_add_float4(const float* x, const float* y, float* output, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int vector_count = size / 4;
    if (index < vector_count) {
        const float4 x4 = reinterpret_cast<const float4*>(x)[index];
        const float4 y4 = reinterpret_cast<const float4*>(y)[index];
        reinterpret_cast<float4*>(output)[index] =
            make_float4(x4.x + y4.x, x4.y + y4.y, x4.z + y4.z, x4.w + y4.w);
    }
    const int tail = vector_count * 4 + index;
    if (index < size - vector_count * 4) {
        output[tail] = x[tail] + y[tail];
    }
}

__global__ void vector_add_half2(
    const __half* x, const __half* y, __half* output, int size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_count = size / 2;
    if (index < pair_count) {
        const __half2 x2 = reinterpret_cast<const __half2*>(x)[index];
        const __half2 y2 = reinterpret_cast<const __half2*>(y)[index];
        reinterpret_cast<__half2*>(output)[index] = __hadd2(x2, y2);
    }
    if (index == 0 && size % 2 != 0) {
        output[size - 1] = __hadd(x[size - 1], y[size - 1]);
    }
}

__global__ void vector_add_bfloat162(
    const __nv_bfloat16* x,
    const __nv_bfloat16* y,
    __nv_bfloat16* output,
    int size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_count = size / 2;
    if (index < pair_count) {
        const __nv_bfloat162 x2 = reinterpret_cast<const __nv_bfloat162*>(x)[index];
        const __nv_bfloat162 y2 = reinterpret_cast<const __nv_bfloat162*>(y)[index];
        reinterpret_cast<__nv_bfloat162*>(output)[index] = __hadd2(x2, y2);
    }
    if (index == 0 && size % 2 != 0) {
        output[size - 1] = __hadd(x[size - 1], y[size - 1]);
    }
}

__global__ void softmax_serial(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) {
        return;
    }
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    float maximum = -FLT_MAX;
    for (int col = 0; col < cols; ++col) {
        maximum = fmaxf(maximum, row_input[col]);
    }
    float denominator = 0.0F;
    for (int col = 0; col < cols; ++col) {
        denominator += expf(row_input[col] - maximum);
    }
    for (int col = 0; col < cols; ++col) {
        row_output[col] = expf(row_input[col] - maximum) / denominator;
    }
}

__global__ void softmax_block(const float* input, float* output, int rows, int cols) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float local_maximum = -FLT_MAX;
    for (int col = lane; col < cols; col += blockDim.x) {
        local_maximum = fmaxf(local_maximum, row_input[col]);
    }
    scratch[lane] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];

    float local_sum = 0.0F;
    for (int col = lane; col < cols; col += blockDim.x) {
        local_sum += expf(row_input[col] - maximum);
    }
    scratch[lane] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (int col = lane; col < cols; col += blockDim.x) {
        row_output[col] = expf(row_input[col] - maximum) / denominator;
    }
}

template <typename Launcher>
std::vector<float> time_launches(
    Launcher launch, int warmup, int samples, int inner, double* cold_start_ms
) {
    const auto cold_start = std::chrono::steady_clock::now();
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto cold_stop = std::chrono::steady_clock::now();
    *cold_start_ms = std::chrono::duration<double, std::milli>(cold_stop - cold_start).count();

    for (int index = 0; index < warmup; ++index) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> measurements;
    measurements.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int index = 0; index < inner; ++index) {
            launch();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        measurements.push_back(elapsed_ms * 1000.0F / static_cast<float>(inner));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return measurements;
}

template <typename T>
void launch_vector_add(
    const T* x,
    const T* y,
    T* output,
    int size,
    const std::string& variant
) {
    constexpr int threads = 256;
    if (variant == "baseline") {
        const int blocks = (size + threads - 1) / threads;
        vector_add_scalar<<<blocks, threads>>>(x, y, output, size);
        return;
    }
    throw std::invalid_argument("optimized vector add is not available for this dtype");
}

template <>
void launch_vector_add<float>(
    const float* x,
    const float* y,
    float* output,
    int size,
    const std::string& variant
) {
    constexpr int threads = 256;
    const int work_items = variant == "optimized" ? (size + 3) / 4 : size;
    const int blocks = (work_items + threads - 1) / threads;
    if (variant == "optimized") {
        vector_add_float4<<<blocks, threads>>>(x, y, output, size);
    } else {
        vector_add_scalar<<<blocks, threads>>>(x, y, output, size);
    }
}

template <>
void launch_vector_add<__half>(
    const __half* x,
    const __half* y,
    __half* output,
    int size,
    const std::string& variant
) {
    constexpr int threads = 256;
    const int work_items = variant == "optimized" ? (size + 1) / 2 : size;
    const int blocks = (work_items + threads - 1) / threads;
    if (variant == "optimized") {
        vector_add_half2<<<blocks, threads>>>(x, y, output, size);
    } else {
        vector_add_scalar<<<blocks, threads>>>(x, y, output, size);
    }
}

template <>
void launch_vector_add<__nv_bfloat16>(
    const __nv_bfloat16* x,
    const __nv_bfloat16* y,
    __nv_bfloat16* output,
    int size,
    const std::string& variant
) {
    constexpr int threads = 256;
    const int work_items = variant == "optimized" ? (size + 1) / 2 : size;
    const int blocks = (work_items + threads - 1) / threads;
    if (variant == "optimized") {
        vector_add_bfloat162<<<blocks, threads>>>(x, y, output, size);
    } else {
        vector_add_scalar<<<blocks, threads>>>(x, y, output, size);
    }
}

void print_result(
    const Options& options,
    const std::string& effective_variant,
    const std::string& shape_json,
    double cold_start_ms,
    const std::vector<float>& samples,
    double max_absolute_error,
    double tolerance,
    double effective_bytes,
    double flops
) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    int driver_version = 0;
    int runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    const double median_us = percentile(samples, 0.5);
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"status\":\"success\",\"operator\":\""
              << options.operation << "\",\"implementation\":\"cuda_cpp\","
              << "\"variant\":\"" << options.variant << "\",\"effective_variant\":\""
              << effective_variant << "\",\"dtype\":\"" << options.dtype
              << "\",\"shape\":" << shape_json << ",\"environment\":{\"gpu\":\""
              << properties.name << "\",\"compute_capability\":[" << properties.major << ','
              << properties.minor << "],\"cuda_driver_api\":" << driver_version
              << ",\"cuda_runtime\":" << runtime_version << "},\"protocol\":{\"samples\":"
              << options.samples << ",\"warmup_calls\":" << options.warmup
              << ",\"inner_iterations\":" << options.inner
              << ",\"timer\":\"cudaEvent\",\"allocation_note\":\"device allocation excluded; "
                 "kernel launch and execution included\"},\"cold_start_ms\":"
              << cold_start_ms << ",\"correctness\":{\"status\":\"passed\","
              << "\"reference\":\"cpu_fp32\",\"atol\":" << tolerance
              << ",\"max_absolute_error\":" << max_absolute_error
              << "},\"steady_state\":{\"latency_us\":{\"median\":" << median_us
              << ",\"p10\":" << percentile(samples, 0.1) << ",\"p90\":"
              << percentile(samples, 0.9) << "}},\"metrics\":{";
    bool emitted = false;
    if (effective_bytes > 0.0) {
        std::cout << "\"effective_bytes\":" << effective_bytes
                  << ",\"effective_gb_per_s\":" << effective_bytes / (median_us * 1.0e3);
        emitted = true;
    }
    if (flops > 0.0) {
        if (emitted) {
            std::cout << ',';
        }
        std::cout << "\"flops\":" << flops << ",\"tflops\":"
                  << flops / (median_us * 1.0e6);
    }
    std::cout << "}}\n";
}

template <typename T>
int run_vector_add(const Options& options) {
    std::vector<T> host_x(options.size);
    std::vector<T> host_y(options.size);
    std::vector<T> host_output(options.size);
    std::vector<T> host_expected(options.size);
    for (int index = 0; index < options.size; ++index) {
        host_x[index] = Numeric<T>::from_float(std::sin(index * 0.001F));
        host_y[index] = Numeric<T>::from_float(std::cos(index * 0.0013F));
        host_expected[index] = Numeric<T>::from_float(
            Numeric<T>::to_float(host_x[index]) + Numeric<T>::to_float(host_y[index])
        );
    }

    T* device_x = nullptr;
    T* device_y = nullptr;
    T* device_output = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(options.size) * sizeof(T);
    CUDA_CHECK(cudaMalloc(&device_x, bytes));
    CUDA_CHECK(cudaMalloc(&device_y, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));
    CUDA_CHECK(cudaMemcpy(device_x, host_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y, host_y.data(), bytes, cudaMemcpyHostToDevice));

    const auto launch = [&]() {
        launch_vector_add(device_x, device_y, device_output, options.size, options.variant);
    };
    double cold_start_ms = 0.0;
    const auto measurements = time_launches(
        launch, options.warmup, options.samples, options.inner, &cold_start_ms
    );
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    double max_absolute_error = 0.0;
    for (int index = 0; index < options.size; ++index) {
        max_absolute_error = std::max(
            max_absolute_error,
            static_cast<double>(std::abs(
                Numeric<T>::to_float(host_output[index]) -
                Numeric<T>::to_float(host_expected[index])
            ))
        );
    }
    const double tolerance = std::is_same_v<T, float> ? 1.0e-6 : 1.0e-3;
    if (max_absolute_error > tolerance) {
        throw std::runtime_error("vector add correctness gate failed");
    }
    CUDA_CHECK(cudaFree(device_x));
    CUDA_CHECK(cudaFree(device_y));
    CUDA_CHECK(cudaFree(device_output));
    print_result(
        options,
        options.variant == "optimized"
            ? (std::is_same_v<T, float> ? "float4" : "packed_pair")
            : "scalar",
        "[" + std::to_string(options.size) + "]",
        cold_start_ms,
        measurements,
        max_absolute_error,
        tolerance,
        3.0 * static_cast<double>(bytes),
        0.0
    );
    return 0;
}

int run_softmax(const Options& options) {
    const int count = options.rows * options.cols;
    std::vector<float> host_input(count);
    std::vector<float> host_output(count);
    std::vector<float> host_expected(count);
    for (int row = 0; row < options.rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < options.cols; ++col) {
            const int index = row * options.cols + col;
            host_input[index] =
                std::sin(index * 0.013F) * 20.0F + std::cos(row * 0.17F);
            maximum = std::max(maximum, host_input[index]);
        }
        double denominator = 0.0;
        for (int col = 0; col < options.cols; ++col) {
            denominator += std::exp(
                static_cast<double>(host_input[row * options.cols + col] - maximum)
            );
        }
        for (int col = 0; col < options.cols; ++col) {
            const int index = row * options.cols + col;
            host_expected[index] = static_cast<float>(
                std::exp(static_cast<double>(host_input[index] - maximum)) / denominator
            );
        }
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&device_input, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));
    const auto launch = [&]() {
        if (options.variant == "baseline") {
            softmax_serial<<<options.rows, 1>>>(
                device_input, device_output, options.rows, options.cols
            );
        } else {
            constexpr int threads = 256;
            softmax_block<<<options.rows, threads, threads * sizeof(float)>>>(
                device_input, device_output, options.rows, options.cols
            );
        }
    };
    double cold_start_ms = 0.0;
    const auto measurements = time_launches(
        launch, options.warmup, options.samples, options.inner, &cold_start_ms
    );
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    double max_absolute_error = 0.0;
    for (int index = 0; index < count; ++index) {
        max_absolute_error = std::max(
            max_absolute_error,
            static_cast<double>(std::abs(host_output[index] - host_expected[index]))
        );
    }
    constexpr double tolerance = 2.0e-5;
    if (max_absolute_error > tolerance) {
        throw std::runtime_error("softmax correctness gate failed");
    }
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    print_result(
        options,
        options.variant == "baseline" ? "one_thread_per_row" : "one_block_per_row",
        "[" + std::to_string(options.rows) + "," + std::to_string(options.cols) + "]",
        cold_start_ms,
        measurements,
        max_absolute_error,
        tolerance,
        2.0 * static_cast<double>(bytes),
        0.0
    );
    return 0;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("every option requires a value");
        }
        const std::string value = argv[++index];
        if (argument == "--operator") {
            options.operation = value;
        } else if (argument == "--variant") {
            options.variant = value;
        } else if (argument == "--dtype") {
            options.dtype = value;
        } else if (argument == "--size") {
            options.size = std::stoi(value);
        } else if (argument == "--rows") {
            options.rows = std::stoi(value);
        } else if (argument == "--cols") {
            options.cols = std::stoi(value);
        } else if (argument == "--samples") {
            options.samples = std::stoi(value);
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value);
        } else if (argument == "--inner") {
            options.inner = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.variant != "baseline" && options.variant != "optimized") {
        throw std::invalid_argument("variant must be baseline or optimized");
    }
    if (options.samples <= 0 || options.warmup < 0 || options.inner <= 0) {
        throw std::invalid_argument("samples/inner must be positive and warmup non-negative");
    }
    if (options.operation == "vector_add") {
        if (options.size <= 0) {
            throw std::invalid_argument("vector_add requires --size > 0");
        }
    } else if (options.operation == "softmax") {
        if (options.rows <= 0 || options.cols <= 0 || options.dtype != "float32") {
            throw std::invalid_argument("softmax requires positive rows/cols and float32");
        }
    } else {
        throw std::invalid_argument("operator must be vector_add or softmax");
    }
    return options;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.operation == "softmax") {
            return run_softmax(options);
        }
        if (options.dtype == "float32") {
            return run_vector_add<float>(options);
        }
        if (options.dtype == "float16") {
            return run_vector_add<__half>(options);
        }
        if (options.dtype == "bfloat16") {
            return run_vector_add<__nv_bfloat16>(options);
        }
        throw std::invalid_argument("vector_add dtype must be float32, float16, or bfloat16");
    } catch (const std::exception& error) {
        std::cerr << "kernel_bench error: " << error.what() << '\n';
        return 2;
    }
}
