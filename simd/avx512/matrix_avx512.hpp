#pragma once
#include <cstddef>

float avx512_dot_float(const float* a_row, const float* b_data, size_t k, size_t n, size_t j);
double avx512_dot_double(const double* a_row, const double* b_data, size_t k, size_t n, size_t j);

void avx512_add_float(const float* a_data, const float* b_data, float* c_data, size_t total);
void avx512_add_double(const double* a_data, const double* b_data, double* c_data, size_t total);
