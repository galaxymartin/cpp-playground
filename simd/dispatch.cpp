#include "dispatch.hpp"
#include "matrix_avx512.hpp"
#include "matrix_sse41.hpp"

#include <atomic>

static std::atomic<SimdLevel> g_level{SimdLevel::AUTO};

void init_simd() {
    if (g_level.load() != SimdLevel::AUTO) return;
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")) {
        g_level.store(SimdLevel::AVX512);
        return;
    }
    if (__builtin_cpu_supports("sse4.2")) {
        g_level.store(SimdLevel::SSE42);
        return;
    }
#endif
    g_level.store(SimdLevel::SCALAR);
}

SimdLevel get_simd_level() { init_simd(); return g_level.load(); }

float simd_dot_float(const float* a_row, const float* b_data, size_t k, size_t n, size_t j, SimdLevel force) {
    SimdLevel lvl = (force == SimdLevel::AUTO) ? get_simd_level() : force;
    if (lvl == SimdLevel::AVX512) return avx512_dot_float(a_row, b_data, k, n, j);
    if (lvl == SimdLevel::SSE42) return sse_dot_float(a_row, b_data, k, n, j);
    float sum = 0.0f;
    for (size_t p = 0; p < k; ++p) sum += a_row[p] * b_data[p * n + j];
    return sum;
}

double simd_dot_double(const double* a_row, const double* b_data, size_t k, size_t n, size_t j, SimdLevel force) {
    SimdLevel lvl = (force == SimdLevel::AUTO) ? get_simd_level() : force;
    if (lvl == SimdLevel::AVX512) return avx512_dot_double(a_row, b_data, k, n, j);
    if (lvl == SimdLevel::SSE42) return sse_dot_double(a_row, b_data, k, n, j);
    double sum = 0.0;
    for (size_t p = 0; p < k; ++p) sum += a_row[p] * b_data[p * n + j];
    return sum;
}

void simd_add_float(const float* a_data, const float* b_data, float* c_data, size_t total, SimdLevel force) {
    SimdLevel lvl = (force == SimdLevel::AUTO) ? get_simd_level() : force;
    if (lvl == SimdLevel::AVX512) { avx512_add_float(a_data, b_data, c_data, total); return; }
    if (lvl == SimdLevel::SSE42) { sse_add_float(a_data, b_data, c_data, total); return; }
    for (size_t i = 0; i < total; ++i) c_data[i] = a_data[i] + b_data[i];
}

void simd_add_double(const double* a_data, const double* b_data, double* c_data, size_t total, SimdLevel force) {
    SimdLevel lvl = (force == SimdLevel::AUTO) ? get_simd_level() : force;
    if (lvl == SimdLevel::AVX512) { avx512_add_double(a_data, b_data, c_data, total); return; }
    if (lvl == SimdLevel::SSE42) { sse_add_double(a_data, b_data, c_data, total); return; }
    for (size_t i = 0; i < total; ++i) c_data[i] = a_data[i] + b_data[i];
}

void simd_sub_float(const float* a_data, const float* b_data, float* c_data, size_t total, SimdLevel force) {
    SimdLevel lvl = (force == SimdLevel::AUTO) ? get_simd_level() : force;
    if (lvl == SimdLevel::AVX512) { avx512_sub_float(a_data, b_data, c_data, total); return; }
    if (lvl == SimdLevel::SSE42) { sse_sub_float(a_data, b_data, c_data, total); return; }
    for (size_t i = 0; i < total; ++i) c_data[i] = a_data[i] - b_data[i];
}

void simd_sub_double(const double* a_data, const double* b_data, double* c_data, size_t total, SimdLevel force) {
    SimdLevel lvl = (force == SimdLevel::AUTO) ? get_simd_level() : force;
    if (lvl == SimdLevel::AVX512) { avx512_sub_double(a_data, b_data, c_data, total); return; }
    if (lvl == SimdLevel::SSE42) { sse_sub_double(a_data, b_data, c_data, total); return; }
    for (size_t i = 0; i < total; ++i) c_data[i] = a_data[i] - b_data[i];
}

// Ensure detection runs at startup
struct SimdAutoInit { SimdAutoInit() { init_simd(); } } simdAutoInit;
