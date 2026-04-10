#pragma once
#include <cstddef>

enum class SimdLevel { AUTO, SCALAR, SSE42, AVX512 };

void init_simd();
SimdLevel get_simd_level();

float simd_dot_float(const float* a_row, const float* b_data, size_t k, size_t n, size_t j, SimdLevel force = SimdLevel::AUTO);
double simd_dot_double(const double* a_row, const double* b_data, size_t k, size_t n, size_t j, SimdLevel force = SimdLevel::AUTO);

void simd_add_float(const float* a_data, const float* b_data, float* c_data, size_t total, SimdLevel force = SimdLevel::AUTO);
void simd_add_double(const double* a_data, const double* b_data, double* c_data, size_t total, SimdLevel force = SimdLevel::AUTO);
