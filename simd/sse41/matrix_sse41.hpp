#pragma once
#include <cstddef>

float sse_dot_float(const float* a_row, const float* b_data, size_t k, size_t n, size_t j);
double sse_dot_double(const double* a_row, const double* b_data, size_t k, size_t n, size_t j);

void sse_add_float(const float* a_data, const float* b_data, float* c_data, size_t total);
void sse_add_double(const double* a_data, const double* b_data, double* c_data, size_t total);
