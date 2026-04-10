#include "matrix_avx512.hpp"
#include <immintrin.h>

float avx512_dot_float(const float* a_row, const float* b_data, size_t k, size_t n, size_t j) {
    __m512 vec_sum = _mm512_setzero_ps();
    size_t p;
    for (p = 0; p + 16 <= k; p += 16) {
        __m512 va = _mm512_loadu_ps(&a_row[p]);
        __m512i indices = _mm512_set_epi32(
            (int)((p + 15) * n + j), (int)((p + 14) * n + j), (int)((p + 13) * n + j), (int)((p + 12) * n + j),
            (int)((p + 11) * n + j), (int)((p + 10) * n + j), (int)((p + 9) * n + j), (int)((p + 8) * n + j),
            (int)((p + 7) * n + j), (int)((p + 6) * n + j), (int)((p + 5) * n + j), (int)((p + 4) * n + j),
            (int)((p + 3) * n + j), (int)((p + 2) * n + j), (int)((p + 1) * n + j), (int)(p * n + j)
        );
        __m512 vb = _mm512_i32gather_ps(indices, b_data, 4);
        vec_sum = _mm512_fmadd_ps(va, vb, vec_sum);
    }
    float temp[16];
    _mm512_storeu_ps(temp, vec_sum);
    float sum = 0.0f;
    for (int t = 0; t < 16; ++t) sum += temp[t];
    for (; p < k; ++p) sum += a_row[p] * b_data[p * n + j];
    return sum;
}

double avx512_dot_double(const double* a_row, const double* b_data, size_t k, size_t n, size_t j) {
    __m512d vec_sum = _mm512_setzero_pd();
    size_t p;
    for (p = 0; p + 8 <= k; p += 8) {
        __m512d va = _mm512_loadu_pd(&a_row[p]);
        __m512i indices = _mm512_set_epi64(
            (long long)((p + 7) * n + j), (long long)((p + 6) * n + j), (long long)((p + 5) * n + j), (long long)((p + 4) * n + j),
            (long long)((p + 3) * n + j), (long long)((p + 2) * n + j), (long long)((p + 1) * n + j), (long long)(p * n + j)
        );
        __m512d vb = _mm512_i64gather_pd(indices, b_data, 8);
        vec_sum = _mm512_fmadd_pd(va, vb, vec_sum);
    }
    double temp[8];
    _mm512_storeu_pd(temp, vec_sum);
    double sum = 0.0;
    for (int t = 0; t < 8; ++t) sum += temp[t];
    for (; p < k; ++p) sum += a_row[p] * b_data[p * n + j];
    return sum;
}

void avx512_add_float(const float* a_data, const float* b_data, float* c_data, size_t total) {
    #pragma omp parallel for
    for (size_t i = 0; i < total; i += 16) {
        if (i + 16 <= total) {
            __m512 va = _mm512_loadu_ps(&a_data[i]);
            __m512 vb = _mm512_loadu_ps(&b_data[i]);
            __m512 vc = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(&c_data[i], vc);
        } else {
            for (size_t j = i; j < total; ++j) c_data[j] = a_data[j] + b_data[j];
        }
    }
}

void avx512_add_double(const double* a_data, const double* b_data, double* c_data, size_t total) {
    #pragma omp parallel for
    for (size_t i = 0; i < total; i += 8) {
        if (i + 8 <= total) {
            __m512d va = _mm512_loadu_pd(&a_data[i]);
            __m512d vb = _mm512_loadu_pd(&b_data[i]);
            __m512d vc = _mm512_add_pd(va, vb);
            _mm512_storeu_pd(&c_data[i], vc);
        } else {
            for (size_t j = i; j < total; ++j) c_data[j] = a_data[j] + b_data[j];
        }
    }
}

void avx512_sub_float(const float* a_data, const float* b_data, float* c_data, size_t total) {
    #pragma omp parallel for
    for (size_t i = 0; i < total; i += 16) {
        if (i + 16 <= total) {
            __m512 va = _mm512_loadu_ps(&a_data[i]);
            __m512 vb = _mm512_loadu_ps(&b_data[i]);
            __m512 vc = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(&c_data[i], vc);
        } else {
            for (size_t j = i; j < total; ++j) c_data[j] = a_data[j] - b_data[j];
        }
    }
}

void avx512_sub_double(const double* a_data, const double* b_data, double* c_data, size_t total) {
    #pragma omp parallel for
    for (size_t i = 0; i < total; i += 8) {
        if (i + 8 <= total) {
            __m512d va = _mm512_loadu_pd(&a_data[i]);
            __m512d vb = _mm512_loadu_pd(&b_data[i]);
            __m512d vc = _mm512_sub_pd(va, vb);
            _mm512_storeu_pd(&c_data[i], vc);
        } else {
            for (size_t j = i; j < total; ++j) c_data[j] = a_data[j] - b_data[j];
        }
    }
}
