#include "matrix.hpp"
#include <type_traits>

// Helper function to check AVX-512 support
static bool has_avx512() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
}

// Helper function to check SSE4.2 support
static bool has_sse42() {
    return __builtin_cpu_supports("sse4.2");
}

// Template implementation
template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows * cols) {}

template<typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> data) {
    m_rows = data.size();
    if (m_rows == 0) m_cols = 0;
    else m_cols = data[0].size();
    m_data.reserve(m_rows * m_cols);
    for (const auto& row : data) {
        if (row.size() != m_cols) throw std::invalid_argument("Inconsistent row sizes");
        m_data.insert(m_data.end(), row.begin(), row.end());
    }
}

template<typename T>
size_t Matrix<T>::rows() const { return m_rows; }

template<typename T>
size_t Matrix<T>::cols() const { return m_cols; }

template<typename T>
T& Matrix<T>::at(size_t i, size_t j) { return m_data[i * m_cols + j]; }

template<typename T>
const T& Matrix<T>::at(size_t i, size_t j) const { return m_data[i * m_cols + j]; }

// Unified multiply function for float and double using AVX-512 if available, SSE4.2 as fallback
template<typename T>
Matrix<T> Matrix<T>::multiply(const Matrix<T>& a, const Matrix<T>& b) {
    if (a.cols() != b.rows()) throw std::invalid_argument("Matrix dimensions don't match");
    size_t m = a.rows(), k = a.cols(), n = b.cols();
    Matrix<T> c(m, n);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            if constexpr (std::is_same_v<T, float>) {
                if (Matrix<T>::s_force_scalar_multiply) {
                    // Scalar fallback
                    for (size_t p = 0; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else if (Matrix<T>::s_force_sse42_multiply && has_sse42()) {
                    __m128 vec_sum = _mm_setzero_ps();
                    size_t p;
                    // Process complete 4-element chunks with SSE4.2
                    for (p = 0; p + 4 <= k; p += 4) {
                        __m128 va = _mm_loadu_ps(&a.m_data[i * k + p]);
                        // Load 4 consecutive elements from b, starting at p*n + j
                        __m128 vb = _mm_setr_ps(
                            b.m_data[p * n + j],
                            b.m_data[(p + 1) * n + j],
                            b.m_data[(p + 2) * n + j],
                            b.m_data[(p + 3) * n + j]
                        );
                        vec_sum = _mm_add_ps(vec_sum, _mm_mul_ps(va, vb));
                    }
                    // Horizontal sum of the 4 elements
                    vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
                    vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
                    sum += _mm_cvtss_f32(vec_sum);
                    // Handle remaining elements with scalar multiplication
                    for (; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else if (has_avx512()) {
                    __m512 vec_sum = _mm512_setzero_ps();
                    size_t p;
                    // Process complete 16-element chunks with AVX-512
                    for (p = 0; p + 16 <= k; p += 16) {
                        __m512 va = _mm512_loadu_ps(&a.m_data[i * k + p]);
                        __m512i indices = _mm512_set_epi32(
                            (p + 15) * n + j, (p + 14) * n + j, (p + 13) * n + j, (p + 12) * n + j,
                            (p + 11) * n + j, (p + 10) * n + j, (p + 9) * n + j, (p + 8) * n + j,
                            (p + 7) * n + j, (p + 6) * n + j, (p + 5) * n + j, (p + 4) * n + j,
                            (p + 3) * n + j, (p + 2) * n + j, (p + 1) * n + j, p * n + j
                        );
                        __m512 vb = _mm512_i32gather_ps(indices, &b.m_data[0], 4);
                        vec_sum = _mm512_fmadd_ps(va, vb, vec_sum);
                    }
                    float temp[16];
                    _mm512_storeu_ps(temp, vec_sum);
                    for (int t = 0; t < 16; ++t) sum += temp[t];
                    // Handle remaining elements with scalar multiplication
                    for (; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else if (has_sse42()) {
                    __m128 vec_sum = _mm_setzero_ps();
                    size_t p;
                    // Process complete 4-element chunks with SSE4.2
                    for (p = 0; p + 4 <= k; p += 4) {
                        __m128 va = _mm_loadu_ps(&a.m_data[i * k + p]);
                        // Load 4 consecutive elements from b, starting at p*n + j
                        __m128 vb = _mm_setr_ps(
                            b.m_data[p * n + j],
                            b.m_data[(p + 1) * n + j],
                            b.m_data[(p + 2) * n + j],
                            b.m_data[(p + 3) * n + j]
                        );
                        vec_sum = _mm_add_ps(vec_sum, _mm_mul_ps(va, vb));
                    }
                    // Horizontal sum of the 4 elements
                    vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
                    vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
                    sum += _mm_cvtss_f32(vec_sum);
                    // Handle remaining elements with scalar multiplication
                    for (; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else {
                    // Scalar fallback
                    for (size_t p = 0; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                }
            } else if constexpr (std::is_same_v<T, double>) {
                if (Matrix<T>::s_force_scalar_multiply) {
                    // Scalar fallback
                    for (size_t p = 0; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else if (Matrix<T>::s_force_sse42_multiply && has_sse42()) {
                    __m128d vec_sum = _mm_setzero_pd();
                    size_t p;
                    // Process complete 2-element chunks with SSE4.2
                    for (p = 0; p + 2 <= k; p += 2) {
                        __m128d va = _mm_loadu_pd(&a.m_data[i * k + p]);
                        // Load 2 consecutive elements from b
                        __m128d vb = _mm_setr_pd(
                            b.m_data[p * n + j],
                            b.m_data[(p + 1) * n + j]
                        );
                        vec_sum = _mm_add_pd(vec_sum, _mm_mul_pd(va, vb));
                    }
                    // Horizontal sum of the 2 elements
                    vec_sum = _mm_hadd_pd(vec_sum, vec_sum);
                    sum += _mm_cvtsd_f64(vec_sum);
                    // Handle remaining elements with scalar multiplication
                    for (; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else if (has_avx512()) {
                    __m512d vec_sum = _mm512_setzero_pd();
                    size_t p;
                    // Process complete 8-element chunks with AVX-512
                    for (p = 0; p + 8 <= k; p += 8) {
                        __m512d va = _mm512_loadu_pd(&a.m_data[i * k + p]);
                        __m512i indices = _mm512_set_epi64(
                            (p + 7) * n + j, (p + 6) * n + j, (p + 5) * n + j, (p + 4) * n + j,
                            (p + 3) * n + j, (p + 2) * n + j, (p + 1) * n + j, p * n + j
                        );
                        __m512d vb = _mm512_i64gather_pd(indices, &b.m_data[0], 8);
                        vec_sum = _mm512_fmadd_pd(va, vb, vec_sum);
                    }
                    double temp[8];
                    _mm512_storeu_pd(temp, vec_sum);
                    for (int t = 0; t < 8; ++t) sum += temp[t];
                    // Handle remaining elements with scalar multiplication
                    for (; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else if (has_sse42()) {
                    __m128d vec_sum = _mm_setzero_pd();
                    size_t p;
                    // Process complete 2-element chunks with SSE4.2
                    for (p = 0; p + 2 <= k; p += 2) {
                        __m128d va = _mm_loadu_pd(&a.m_data[i * k + p]);
                        // Load 2 consecutive elements from b
                        __m128d vb = _mm_setr_pd(
                            b.m_data[p * n + j],
                            b.m_data[(p + 1) * n + j]
                        );
                        vec_sum = _mm_add_pd(vec_sum, _mm_mul_pd(va, vb));
                    }
                    // Horizontal sum of the 2 elements
                    vec_sum = _mm_hadd_pd(vec_sum, vec_sum);
                    sum += _mm_cvtsd_f64(vec_sum);
                    // Handle remaining elements with scalar multiplication
                    for (; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                } else {
                    // Scalar fallback
                    for (size_t p = 0; p < k; ++p) {
                        sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                    }
                }
            } else {
                // Fallback for other types
                for (size_t p = 0; p < k; ++p) {
                    sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                }
            }
            c.at(i, j) = sum;
        }
    }
    return c;
}

// Unified add function for float and double using AVX-512 if available, SSE4.2 as fallback
template<typename T>
Matrix<T> Matrix<T>::add(const Matrix<T>& a, const Matrix<T>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    size_t total = a.rows() * a.cols();
    Matrix<T> c(a.rows(), a.cols());
    
    if constexpr (std::is_same_v<T, float>) {
        if (Matrix<T>::s_force_scalar_add) {
            // Scalar fallback
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) {
                c.m_data[i] = a.m_data[i] + b.m_data[i];
            }
        } else if (Matrix<T>::s_force_sse42_add && has_sse42()) {
            // SSE4.2 addition
            #pragma omp parallel for
            for (size_t i = 0; i < total; i += 4) {
                if (i + 4 <= total) {
                    __m128 va = _mm_loadu_ps(&a.m_data[i]);
                    __m128 vb = _mm_loadu_ps(&b.m_data[i]);
                    __m128 vc = _mm_add_ps(va, vb);
                    _mm_storeu_ps(&c.m_data[i], vc);
                } else {
                    for (size_t j = i; j < total; ++j) {
                        c.m_data[j] = a.m_data[j] + b.m_data[j];
                    }
                }
            }
        } else if (has_avx512()) {
            // AVX-512 addition
            #pragma omp parallel for
            for (size_t i = 0; i < total; i += 16) {
                if (i + 16 <= total) {
                    __m512 va = _mm512_loadu_ps(&a.m_data[i]);
                    __m512 vb = _mm512_loadu_ps(&b.m_data[i]);
                    __m512 vc = _mm512_add_ps(va, vb);
                    _mm512_storeu_ps(&c.m_data[i], vc);
                } else {
                    for (size_t j = i; j < total; ++j) {
                        c.m_data[j] = a.m_data[j] + b.m_data[j];
                    }
                }
            }
        } else if (has_sse42()) {
            // SSE4.2 addition
            #pragma omp parallel for
            for (size_t i = 0; i < total; i += 4) {
                if (i + 4 <= total) {
                    __m128 va = _mm_loadu_ps(&a.m_data[i]);
                    __m128 vb = _mm_loadu_ps(&b.m_data[i]);
                    __m128 vc = _mm_add_ps(va, vb);
                    _mm_storeu_ps(&c.m_data[i], vc);
                } else {
                    for (size_t j = i; j < total; ++j) {
                        c.m_data[j] = a.m_data[j] + b.m_data[j];
                    }
                }
            }
        } else {
            // Scalar fallback
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) {
                c.m_data[i] = a.m_data[i] + b.m_data[i];
            }
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (Matrix<T>::s_force_scalar_add) {
            // Scalar fallback
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) {
                c.m_data[i] = a.m_data[i] + b.m_data[i];
            }
        } else if (Matrix<T>::s_force_sse42_add && has_sse42()) {
            // SSE4.2 addition
            #pragma omp parallel for
            for (size_t i = 0; i < total; i += 2) {
                if (i + 2 <= total) {
                    __m128d va = _mm_loadu_pd(&a.m_data[i]);
                    __m128d vb = _mm_loadu_pd(&b.m_data[i]);
                    __m128d vc = _mm_add_pd(va, vb);
                    _mm_storeu_pd(&c.m_data[i], vc);
                } else {
                    for (size_t j = i; j < total; ++j) {
                        c.m_data[j] = a.m_data[j] + b.m_data[j];
                    }
                }
            }
        } else if (has_avx512()) {
            // AVX-512 addition
            #pragma omp parallel for
            for (size_t i = 0; i < total; i += 8) {
                if (i + 8 <= total) {
                    __m512d va = _mm512_loadu_pd(&a.m_data[i]);
                    __m512d vb = _mm512_loadu_pd(&b.m_data[i]);
                    __m512d vc = _mm512_add_pd(va, vb);
                    _mm512_storeu_pd(&c.m_data[i], vc);
                } else {
                    for (size_t j = i; j < total; ++j) {
                        c.m_data[j] = a.m_data[j] + b.m_data[j];
                    }
                }
            }
        } else if (has_sse42()) {
            // SSE4.2 addition
            #pragma omp parallel for
            for (size_t i = 0; i < total; i += 2) {
                if (i + 2 <= total) {
                    __m128d va = _mm_loadu_pd(&a.m_data[i]);
                    __m128d vb = _mm_loadu_pd(&b.m_data[i]);
                    __m128d vc = _mm_add_pd(va, vb);
                    _mm_storeu_pd(&c.m_data[i], vc);
                } else {
                    for (size_t j = i; j < total; ++j) {
                        c.m_data[j] = a.m_data[j] + b.m_data[j];
                    }
                }
            }
        } else {
            // Scalar fallback
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) {
                c.m_data[i] = a.m_data[i] + b.m_data[i];
            }
        }
    } else {
        // Fallback for other types
        #pragma omp parallel for
        for (size_t i = 0; i < total; ++i) {
            c.m_data[i] = a.m_data[i] + b.m_data[i];
        }
    }
    return c;
}

// Explicit instantiations for multiply and add
template Matrix<float> Matrix<float>::multiply(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> Matrix<double>::multiply(const Matrix<double>&, const Matrix<double>&);
template Matrix<float> Matrix<float>::add(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> Matrix<double>::add(const Matrix<double>&, const Matrix<double>&);

// Explicit instantiations for basic methods
template Matrix<float>::Matrix(size_t rows, size_t cols);
template Matrix<float>::Matrix(std::vector<std::vector<float>> data);
template size_t Matrix<float>::rows() const;
template size_t Matrix<float>::cols() const;
template float& Matrix<float>::at(size_t i, size_t j);
template const float& Matrix<float>::at(size_t i, size_t j) const;

template Matrix<double>::Matrix(size_t rows, size_t cols);
template Matrix<double>::Matrix(std::vector<std::vector<double>> data);
template size_t Matrix<double>::rows() const;
template size_t Matrix<double>::cols() const;
template double& Matrix<double>::at(size_t i, size_t j);
template const double& Matrix<double>::at(size_t i, size_t j) const;