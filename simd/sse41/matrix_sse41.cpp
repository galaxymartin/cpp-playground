#include "matrix_sse41.hpp"
#include <immintrin.h>

float sse_dot_float(const float* a_row, const float* b_data, size_t k, size_t n, size_t j) {
    __m128 vec_sum = _mm_setzero_ps();
    size_t p;
    for (p = 0; p + 4 <= k; p += 4) {
        __m128 va = _mm_loadu_ps(&a_row[p]);
        __m128 vb = _mm_setr_ps(
            b_data[p * n + j], b_data[(p + 1) * n + j], b_data[(p + 2) * n + j], b_data[(p + 3) * n + j]
        );
        vec_sum = _mm_add_ps(vec_sum, _mm_mul_ps(va, vb));
    }
    vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
    vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
    float sum = _mm_cvtss_f32(vec_sum);
    for (; p < k; ++p) sum += a_row[p] * b_data[p * n + j];
    return sum;
}

double sse_dot_double(const double* a_row, const double* b_data, size_t k, size_t n, size_t j) {
    __m128d vec_sum = _mm_setzero_pd();
    size_t p;
    for (p = 0; p + 2 <= k; p += 2) {
        __m128d va = _mm_loadu_pd(&a_row[p]);
        __m128d vb = _mm_setr_pd(b_data[p * n + j], b_data[(p + 1) * n + j]);
        vec_sum = _mm_add_pd(vec_sum, _mm_mul_pd(va, vb));
    }
    vec_sum = _mm_hadd_pd(vec_sum, vec_sum);
    double sum = _mm_cvtsd_f64(vec_sum);
    for (; p < k; ++p) sum += a_row[p] * b_data[p * n + j];
    return sum;
}

void sse_add_float(const float* a_data, const float* b_data, float* c_data, size_t total) {
    #pragma omp parallel for
    for (size_t i = 0; i < total; i += 4) {
        if (i + 4 <= total) {
            __m128 va = _mm_loadu_ps(&a_data[i]);
            __m128 vb = _mm_loadu_ps(&b_data[i]);
            __m128 vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(&c_data[i], vc);
        } else {
            for (size_t j = i; j < total; ++j) c_data[j] = a_data[j] + b_data[j];
        }
    }
}

void sse_add_double(const double* a_data, const double* b_data, double* c_data, size_t total) {
    #pragma omp parallel for
    for (size_t i = 0; i < total; i += 2) {
        if (i + 2 <= total) {
            __m128d va = _mm_loadu_pd(&a_data[i]);
            __m128d vb = _mm_loadu_pd(&b_data[i]);
            __m128d vc = _mm_add_pd(va, vb);
            _mm_storeu_pd(&c_data[i], vc);
        } else {
            for (size_t j = i; j < total; ++j) c_data[j] = a_data[j] + b_data[j];
        }
    }
}
