#pragma once

#include <vector>
#include <cstddef>
#include <immintrin.h>
#include <stdexcept>

template<typename T>
class Matrix {
private:
    static inline bool s_force_scalar_multiply = false;
    static inline bool s_force_sse42_multiply = false;
    static inline bool s_force_scalar_add = false;
    static inline bool s_force_sse42_add = false;
    
public:
    Matrix(size_t rows, size_t cols);
    Matrix(std::vector<std::vector<T>> data);
    size_t rows() const;
    size_t cols() const;
    T& at(size_t i, size_t j);
    const T& at(size_t i, size_t j) const;
    static Matrix multiply(const Matrix& a, const Matrix& b);
    static Matrix add(const Matrix& a, const Matrix& b);
    
    static void setForceScalarMultiply(bool force) { s_force_scalar_multiply = force; }
    static void setForceSSE42Multiply(bool force) { s_force_sse42_multiply = force; }
    static void setForceScalarAdd(bool force) { s_force_scalar_add = force; }
    static void setForceSSE42Add(bool force) { s_force_sse42_add = force; }

    size_t m_rows, m_cols;
    std::vector<T> m_data;
};

// Type aliases for convenience
using MatrixF = Matrix<float>;
using MatrixD = Matrix<double>;

// Operator overloads
template<typename T>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b) {
    return Matrix<T>::add(a, b);
}

template<typename T>
Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
    return Matrix<T>::multiply(a, b);
}