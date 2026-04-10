#include "matrix.hpp"
#include <type_traits>

#include "simd/dispatch.hpp"

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

// Unified multiply function for float and double using runtime SIMD dispatch
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
                    for (size_t p = 0; p < k; ++p) sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                } else if (Matrix<T>::s_force_sse42_multiply) {
                    sum = simd_dot_float(&a.m_data[i * k], &b.m_data[0], k, n, j, SimdLevel::SSE42);
                } else {
                    sum = simd_dot_float(&a.m_data[i * k], &b.m_data[0], k, n, j, SimdLevel::AUTO);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                if (Matrix<T>::s_force_scalar_multiply) {
                    for (size_t p = 0; p < k; ++p) sum += a.m_data[i * k + p] * b.m_data[p * n + j];
                } else if (Matrix<T>::s_force_sse42_multiply) {
                    sum = simd_dot_double(&a.m_data[i * k], &b.m_data[0], k, n, j, SimdLevel::SSE42);
                } else {
                    sum = simd_dot_double(&a.m_data[i * k], &b.m_data[0], k, n, j, SimdLevel::AUTO);
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
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) c.m_data[i] = a.m_data[i] + b.m_data[i];
        } else if (Matrix<T>::s_force_sse42_add) {
            simd_add_float(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::SSE42);
        } else {
            simd_add_float(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::AUTO);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (Matrix<T>::s_force_scalar_add) {
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) c.m_data[i] = a.m_data[i] + b.m_data[i];
        } else if (Matrix<T>::s_force_sse42_add) {
            simd_add_double(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::SSE42);
        } else {
            simd_add_double(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::AUTO);
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

template<typename T>
Matrix<T> Matrix<T>::sub(const Matrix<T>& a, const Matrix<T>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    size_t total = a.rows() * a.cols();
    Matrix<T> c(a.rows(), a.cols());
    if constexpr (std::is_same_v<T, float>) {
        if (Matrix<T>::s_force_scalar_sub) {
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) c.m_data[i] = a.m_data[i] - b.m_data[i];
        } else if (Matrix<T>::s_force_sse42_sub) {
            simd_sub_float(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::SSE42);
        } else {
            simd_sub_float(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::AUTO);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (Matrix<T>::s_force_scalar_sub) {
            #pragma omp parallel for
            for (size_t i = 0; i < total; ++i) c.m_data[i] = a.m_data[i] - b.m_data[i];
        } else if (Matrix<T>::s_force_sse42_sub) {
            simd_sub_double(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::SSE42);
        } else {
            simd_sub_double(&a.m_data[0], &b.m_data[0], &c.m_data[0], total, SimdLevel::AUTO);
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < total; ++i) c.m_data[i] = a.m_data[i] - b.m_data[i];
    }
    return c;
}

// Explicit instantiations for multiply and add
template Matrix<float> Matrix<float>::multiply(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> Matrix<double>::multiply(const Matrix<double>&, const Matrix<double>&);
template Matrix<float> Matrix<float>::add(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> Matrix<double>::add(const Matrix<double>&, const Matrix<double>&);
template Matrix<float> Matrix<float>::sub(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> Matrix<double>::sub(const Matrix<double>&, const Matrix<double>&);

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