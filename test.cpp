#include <algorithm>
#include <type_traits>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "mock_turtle.hpp"
#include "mock_worklist_adapter.hpp"
#include "worklist_port.hpp"
#include "painter.hpp"
#include "matrix.hpp"

using ::testing::AtLeast;
using ::testing::Return;

// Helper function to fill matrix using AVX512
template<typename T>
void fillMatrixAVX512(Matrix<T>& mat, int mod, T factor) {
  size_t total = mat.rows() * mat.cols();
  for (size_t i = 0; i < total; ++i) {
    mat.m_data[i] = static_cast<T>((i % mod)) * factor;
  }
}

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(PainterTest, CanDrawSomething) {
  MockTurtle turtle;
  EXPECT_CALL(turtle, PenDown()).Times(AtLeast(1));
  EXPECT_CALL(turtle, GoTo(4,5)).Times(AtLeast(1));

  Painter painter(&turtle);

  EXPECT_TRUE(painter.DrawCircle(0,0,10));
}

TEST(ServiceTest, BasicCall) {
  MockWorklistAdapter adapter;
  EXPECT_CALL(adapter, getWorklist(123)).Times(1).WillOnce(Return("test"));

  WorklistPort port(&adapter);
  EXPECT_EQ(port.getWorklistOrdered(123), "test");
}

TEST(MatrixTest, Multiply) {
  Matrix<float> a({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}});
  Matrix<float> b({{1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1},
                   {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}});
  Matrix<float> c = Matrix<float>::multiply(a, b);
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  // Compute expected: a is 2x16, b is 16x2
  // c[0][0] = sum a[0][p] * b[p][0] for p=0..15
  // b[p][0] = 1 if p even, 0 if odd
  // So sum a[0][even] = 1+3+5+7+9+11+13+15 = 64
  // Similarly c[0][1] = sum a[0][odd] = 2+4+6+8+10+12+14+16 = 72
  // c[1][0] = 2+4+6+8+10+12+14+16+18 = 90 (since a[1] = 2 to 17)
  // a[1] = 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
  // even p: 2,4,6,8,10,12,14,16 sum=72
  // odd p: 3,5,7,9,11,13,15,17 sum=80
  EXPECT_NEAR(c.at(0, 0), 64.0f, 1e-5f);
  EXPECT_NEAR(c.at(0, 1), 72.0f, 1e-5f);
  EXPECT_NEAR(c.at(1, 0), 72.0f, 1e-5f);
  EXPECT_NEAR(c.at(1, 1), 80.0f, 1e-5f);
}


TEST(MatrixTest, MultiplyConsistency) {
  // Multiple runs should give identical results
  Matrix<float> a({{1, 2}, {3, 4}});
  Matrix<float> b({{5, 6}, {7, 8}});
  
  Matrix<float> c1 = Matrix<float>::multiply(a, b);
  Matrix<float> c2 = Matrix<float>::multiply(a, b);
  
  for (size_t i = 0; i < c1.rows(); ++i) {
    for (size_t j = 0; j < c1.cols(); ++j) {
      EXPECT_FLOAT_EQ(c1.at(i, j), c2.at(i, j));
    }
  }
}

TEST(MatrixDoubleTest, Multiply) {
  Matrix<double> a({{1, 2, 3, 4, 5, 6, 7, 8},
                    {2, 3, 4, 5, 6, 7, 8, 9}});
  Matrix<double> b({{1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}});
  Matrix<double> c = Matrix<double>::multiply(a, b);
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  // a is 2x8, b is 8x2
  // c[0][0] = sum a[0][p] * b[p][0] for p=0..7
  // b[p][0] = 1 if p even, 0 if odd
  // sum a[0][even] = 1+3+5+7 = 16
  // c[0][1] = sum a[0][odd] = 2+4+6+8 = 20
  // c[1][0] = sum a[1][even] = 2+4+6+8 = 20
  // c[1][1] = sum a[1][odd] = 3+5+7+9 = 24
  EXPECT_NEAR(c.at(0, 0), 16.0, 1e-10);
  EXPECT_NEAR(c.at(0, 1), 20.0, 1e-10);
  EXPECT_NEAR(c.at(1, 0), 20.0, 1e-10);
  EXPECT_NEAR(c.at(1, 1), 24.0, 1e-10);
}

TEST(MatrixDoubleTest, MultiplyConsistency) {
  // Multiple runs should give identical results
  Matrix<double> a({{1, 2}, {3, 4}});
  Matrix<double> b({{5, 6}, {7, 8}});
  
  Matrix<double> c1 = Matrix<double>::multiply(a, b);
  Matrix<double> c2 = Matrix<double>::multiply(a, b);
  
  for (size_t i = 0; i < c1.rows(); ++i) {
    for (size_t j = 0; j < c1.cols(); ++j) {
      EXPECT_DOUBLE_EQ(c1.at(i, j), c2.at(i, j));
    }
  }
}

TEST(MatrixTest, AddBasic) {
  Matrix<float> a({{1, 2}, {3, 4}});
  Matrix<float> b({{5, 6}, {7, 8}});
  Matrix<float> c = Matrix<float>::add(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_NEAR(c.at(0, 0), 6.0f, 1e-5f);   // 1 + 5
  EXPECT_NEAR(c.at(0, 1), 8.0f, 1e-5f);   // 2 + 6
  EXPECT_NEAR(c.at(1, 0), 10.0f, 1e-5f);  // 3 + 7
  EXPECT_NEAR(c.at(1, 1), 12.0f, 1e-5f);  // 4 + 8
}


TEST(MatrixTest, AddDimensionMismatch) {
  Matrix<float> a({{1, 2}, {3, 4}});
  Matrix<float> b({{5, 6}, {7, 8}, {9, 10}});  // Different rows
  
  EXPECT_THROW(Matrix<float>::add(a, b), std::invalid_argument);
}

TEST(MatrixTest, OperatorOverloads) {
  Matrix<float> a({{1, 2}, {3, 4}});
  Matrix<float> b({{5, 6}, {7, 8}});
  
  // Test operator+
  Matrix<float> c_add = a + b;
  Matrix<float> expected_add = Matrix<float>::add(a, b);
  for (size_t i = 0; i < c_add.rows(); ++i) {
    for (size_t j = 0; j < c_add.cols(); ++j) {
      EXPECT_FLOAT_EQ(c_add.at(i, j), expected_add.at(i, j));
    }
  }
  
  // Test operator*
  Matrix<float> c_mul = a * b;
  Matrix<float> expected_mul = Matrix<float>::multiply(a, b);
  for (size_t i = 0; i < c_mul.rows(); ++i) {
    for (size_t j = 0; j < c_mul.cols(); ++j) {
      EXPECT_FLOAT_EQ(c_mul.at(i, j), expected_mul.at(i, j));
    }
  }
}

TEST(MatrixDoubleTest, AddBasic) {
  Matrix<double> a({{1, 2}, {3, 4}});
  Matrix<double> b({{5, 6}, {7, 8}});
  Matrix<double> c = Matrix<double>::add(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_NEAR(c.at(0, 0), 6.0, 1e-10);   // 1 + 5
  EXPECT_NEAR(c.at(0, 1), 8.0, 1e-10);   // 2 + 6
  EXPECT_NEAR(c.at(1, 0), 10.0, 1e-10);  // 3 + 7
  EXPECT_NEAR(c.at(1, 1), 12.0, 1e-10);  // 4 + 8
}

TEST(MatrixDoubleTest, AddDimensionMismatch) {
  Matrix<double> a({{1, 2}, {3, 4}});
  Matrix<double> b({{5, 6}, {7, 8}, {9, 10}});  // Different rows
  
  EXPECT_THROW(Matrix<double>::add(a, b), std::invalid_argument);
}

TEST(MatrixTest, AddLargeMatrices) {
  // Create ~200MB matrices in memory (float: 4 bytes per element, 7245 x 7245 = ~200MB per matrix)
  size_t rows = 7245, cols = 7245;
  
  // Generate matrix A
  Matrix<float> a(rows, cols);
  fillMatrixAVX512(a, 1000, 0.1f);
  
  // Generate matrix B
  Matrix<float> b(rows, cols);
  fillMatrixAVX512(b, 500, 0.2f);
  
  EXPECT_EQ(a.rows(), rows);
  EXPECT_EQ(a.cols(), cols);
  EXPECT_EQ(b.rows(), rows);
  EXPECT_EQ(b.cols(), cols);
  
  // Perform addition
  Matrix<float> c = Matrix<float>::add(a, b);
  EXPECT_EQ(c.rows(), rows);
  EXPECT_EQ(c.cols(), cols);
  
  // Verify some results (sample every 10000th element to avoid slow full verification)
  for (size_t i = 0; i < rows * cols; i += 10000) {
    EXPECT_NEAR(c.m_data[i], a.m_data[i] + b.m_data[i], 1e-4f);
  }
}

TEST(MatrixDoubleTest, MultiplyLargeMatrices) {
  // Create ~200MB matrices in memory for multiplication (double precision)
  size_t m = 5120, k = 5120, n = 5120;  // A is m×k, B is k×n, C is m×n
  
  // Generate matrix A (5120 x 5120)
  Matrix<double> a(m, k);
  fillMatrixAVX512(a, 100, 0.01);
  
  // Generate matrix B (5120 x 5120)
  Matrix<double> b(k, n);
  fillMatrixAVX512(b, 50, 0.02);
  
  EXPECT_EQ(a.rows(), m);
  EXPECT_EQ(a.cols(), k);
  EXPECT_EQ(b.rows(), k);
  EXPECT_EQ(b.cols(), n);
  
  // Perform multiplication
  Matrix<double> c = Matrix<double>::multiply(a, b);
  EXPECT_EQ(c.rows(), m);
  EXPECT_EQ(c.cols(), n);
  
  // Verify result has non-zero values
  bool hasNonZero = false;
  for (size_t i = 0; i < c.m_data.size(); ++i) {
    if (c.m_data[i] != 0.0) {
      hasNonZero = true;
      break;
    }
  }
  EXPECT_TRUE(hasNonZero);
}

// Tests for scalar multiplication (non-AVX512 path)
TEST(MatrixTest, MultiplyScalar) {
  Matrix<float>::setForceScalarMultiply(true);
  
  Matrix<float> a(2, 3);
  Matrix<float> b(3, 2);
  
  // Initialize matrices
  a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f; a.at(0, 2) = 3.0f;
  a.at(1, 0) = 4.0f; a.at(1, 1) = 5.0f; a.at(1, 2) = 6.0f;
  
  b.at(0, 0) = 7.0f; b.at(0, 1) = 8.0f;
  b.at(1, 0) = 9.0f; b.at(1, 1) = 10.0f;
  b.at(2, 0) = 11.0f; b.at(2, 1) = 12.0f;
  
  Matrix<float> c = Matrix<float>::multiply(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_FLOAT_EQ(c.at(0, 0), 58.0f);  // 1*7 + 2*9 + 3*11
  EXPECT_FLOAT_EQ(c.at(0, 1), 64.0f);  // 1*8 + 2*10 + 3*12
  EXPECT_FLOAT_EQ(c.at(1, 0), 139.0f); // 4*7 + 5*9 + 6*11
  EXPECT_FLOAT_EQ(c.at(1, 1), 154.0f); // 4*8 + 5*10 + 6*12
  
  Matrix<float>::setForceScalarMultiply(false);
}

TEST(MatrixDoubleTest, MultiplyScalar) {
  Matrix<double>::setForceScalarMultiply(true);
  
  Matrix<double> a(2, 3);
  Matrix<double> b(3, 2);
  
  // Initialize matrices
  a.at(0, 0) = 1.0; a.at(0, 1) = 2.0; a.at(0, 2) = 3.0;
  a.at(1, 0) = 4.0; a.at(1, 1) = 5.0; a.at(1, 2) = 6.0;
  
  b.at(0, 0) = 7.0; b.at(0, 1) = 8.0;
  b.at(1, 0) = 9.0; b.at(1, 1) = 10.0;
  b.at(2, 0) = 11.0; b.at(2, 1) = 12.0;
  
  Matrix<double> c = Matrix<double>::multiply(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_DOUBLE_EQ(c.at(0, 0), 58.0);  // 1*7 + 2*9 + 3*11
  EXPECT_DOUBLE_EQ(c.at(0, 1), 64.0);  // 1*8 + 2*10 + 3*12
  EXPECT_DOUBLE_EQ(c.at(1, 0), 139.0); // 4*7 + 5*9 + 6*11
  EXPECT_DOUBLE_EQ(c.at(1, 1), 154.0); // 4*8 + 5*10 + 6*12
  
  Matrix<double>::setForceScalarMultiply(false);
}

TEST(MatrixTest, MultiplyScalarConsistency) {
  Matrix<float>::setForceScalarMultiply(true);
  
  // Test that scalar and SIMD produce identical results
  Matrix<float> a(4, 4);
  Matrix<float> b(4, 4);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      a.at(i, j) = static_cast<float>(i * 4 + j + 1);
      b.at(i, j) = static_cast<float>((i * 4 + j + 1) * 2);
    }
  }
  
  Matrix<float> c_scalar = Matrix<float>::multiply(a, b);
  
  // Reset to allow SIMD
  Matrix<float>::setForceScalarMultiply(false);
  Matrix<float> c_simd = Matrix<float>::multiply(a, b);
  
  // Compare results
  EXPECT_EQ(c_scalar.rows(), c_simd.rows());
  EXPECT_EQ(c_scalar.cols(), c_simd.cols());
  for (size_t i = 0; i < c_scalar.rows(); ++i) {
    for (size_t j = 0; j < c_scalar.cols(); ++j) {
      EXPECT_FLOAT_EQ(c_scalar.at(i, j), c_simd.at(i, j));
    }
  }
}

TEST(MatrixDoubleTest, MultiplyScalarConsistency) {
  Matrix<double>::setForceScalarMultiply(true);
  
  // Test that scalar and SIMD produce identical results
  Matrix<double> a(4, 4);
  Matrix<double> b(4, 4);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      a.at(i, j) = static_cast<double>(i * 4 + j + 1);
      b.at(i, j) = static_cast<double>((i * 4 + j + 1) * 2);
    }
  }
  
  Matrix<double> c_scalar = Matrix<double>::multiply(a, b);
  
  // Reset to allow SIMD
  Matrix<double>::setForceScalarMultiply(false);
  Matrix<double> c_simd = Matrix<double>::multiply(a, b);
  
  // Compare results
  EXPECT_EQ(c_scalar.rows(), c_simd.rows());
  EXPECT_EQ(c_scalar.cols(), c_simd.cols());
  for (size_t i = 0; i < c_scalar.rows(); ++i) {
    for (size_t j = 0; j < c_scalar.cols(); ++j) {
      EXPECT_DOUBLE_EQ(c_scalar.at(i, j), c_simd.at(i, j));
    }
  }
}

// Tests for SSE4.2 multiplication (fallback SIMD path)
TEST(MatrixTest, MultiplySSE42) {
  Matrix<float>::setForceSSE42Multiply(true);
  
  Matrix<float> a(2, 4);  // Use multiple of 4 for better SSE testing
  Matrix<float> b(4, 2);
  
  // Initialize matrices
  a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f; a.at(0, 2) = 3.0f; a.at(0, 3) = 4.0f;
  a.at(1, 0) = 5.0f; a.at(1, 1) = 6.0f; a.at(1, 2) = 7.0f; a.at(1, 3) = 8.0f;
  
  b.at(0, 0) = 9.0f; b.at(0, 1) = 10.0f;
  b.at(1, 0) = 11.0f; b.at(1, 1) = 12.0f;
  b.at(2, 0) = 13.0f; b.at(2, 1) = 14.0f;
  b.at(3, 0) = 15.0f; b.at(3, 1) = 16.0f;
  
  Matrix<float> c = Matrix<float>::multiply(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_FLOAT_EQ(c.at(0, 0), 1*9 + 2*11 + 3*13 + 4*15);  // 130
  EXPECT_FLOAT_EQ(c.at(0, 1), 1*10 + 2*12 + 3*14 + 4*16); // 140
  EXPECT_FLOAT_EQ(c.at(1, 0), 5*9 + 6*11 + 7*13 + 8*15);  // 314
  EXPECT_FLOAT_EQ(c.at(1, 1), 5*10 + 6*12 + 7*14 + 8*16); // 348
  
  Matrix<float>::setForceSSE42Multiply(false);
}

TEST(MatrixDoubleTest, MultiplySSE42) {
  Matrix<double>::setForceSSE42Multiply(true);
  
  Matrix<double> a(2, 4);  // Use multiple of 4 for better SSE testing
  Matrix<double> b(4, 2);
  
  // Initialize matrices
  a.at(0, 0) = 1.0; a.at(0, 1) = 2.0; a.at(0, 2) = 3.0; a.at(0, 3) = 4.0;
  a.at(1, 0) = 5.0; a.at(1, 1) = 6.0; a.at(1, 2) = 7.0; a.at(1, 3) = 8.0;
  
  b.at(0, 0) = 9.0; b.at(0, 1) = 10.0;
  b.at(1, 0) = 11.0; b.at(1, 1) = 12.0;
  b.at(2, 0) = 13.0; b.at(2, 1) = 14.0;
  b.at(3, 0) = 15.0; b.at(3, 1) = 16.0;
  
  Matrix<double> c = Matrix<double>::multiply(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_DOUBLE_EQ(c.at(0, 0), 1*9 + 2*11 + 3*13 + 4*15);  // 130
  EXPECT_DOUBLE_EQ(c.at(0, 1), 1*10 + 2*12 + 3*14 + 4*16); // 140
  EXPECT_DOUBLE_EQ(c.at(1, 0), 5*9 + 6*11 + 7*13 + 8*15);  // 314
  EXPECT_DOUBLE_EQ(c.at(1, 1), 5*10 + 6*12 + 7*14 + 8*16); // 348
  
  Matrix<double>::setForceSSE42Multiply(false);
}

TEST(MatrixTest, MultiplySSE42Consistency) {
  Matrix<float>::setForceSSE42Multiply(true);
  
  // Test that SSE4.2 and AVX-512 produce identical results (if AVX-512 available)
  Matrix<float> a(4, 8);  // Use dimensions that work well with both
  Matrix<float> b(8, 4);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      a.at(i, j) = static_cast<float>(i * 8 + j + 1);
    }
  }
  for (size_t i = 0; i < 8; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      b.at(i, j) = static_cast<float>((i * 4 + j + 1) * 2);
    }
  }
  
  Matrix<float> c_sse42 = Matrix<float>::multiply(a, b);
  
  // Reset to allow AVX-512
  Matrix<float>::setForceSSE42Multiply(false);
  Matrix<float> c_avx512 = Matrix<float>::multiply(a, b);
  
  // Compare results
  EXPECT_EQ(c_sse42.rows(), c_avx512.rows());
  EXPECT_EQ(c_sse42.cols(), c_avx512.cols());
  for (size_t i = 0; i < c_sse42.rows(); ++i) {
    for (size_t j = 0; j < c_sse42.cols(); ++j) {
      EXPECT_FLOAT_EQ(c_sse42.at(i, j), c_avx512.at(i, j));
    }
  }
}

TEST(MatrixDoubleTest, MultiplySSE42Consistency) {
  Matrix<double>::setForceSSE42Multiply(true);
  
  // Test that SSE4.2 and AVX-512 produce identical results (if AVX-512 available)
  Matrix<double> a(4, 8);  // Use dimensions that work well with both
  Matrix<double> b(8, 4);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      a.at(i, j) = static_cast<double>(i * 8 + j + 1);
    }
  }
  for (size_t i = 0; i < 8; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      b.at(i, j) = static_cast<double>((i * 4 + j + 1) * 2);
    }
  }
  
  Matrix<double> c_sse42 = Matrix<double>::multiply(a, b);
  
  // Reset to allow AVX-512
  Matrix<double>::setForceSSE42Multiply(false);
  Matrix<double> c_avx512 = Matrix<double>::multiply(a, b);
  
  // Compare results
  EXPECT_EQ(c_sse42.rows(), c_avx512.rows());
  EXPECT_EQ(c_sse42.cols(), c_avx512.cols());
  for (size_t i = 0; i < c_sse42.rows(); ++i) {
    for (size_t j = 0; j < c_sse42.cols(); ++j) {
      EXPECT_DOUBLE_EQ(c_sse42.at(i, j), c_avx512.at(i, j));
    }
  }
}

// Tests for scalar addition (non-SIMD path)
TEST(MatrixTest, AddScalar) {
  Matrix<float>::setForceScalarAdd(true);
  
  Matrix<float> a(2, 4);
  Matrix<float> b(2, 4);
  
  // Initialize matrices
  a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f; a.at(0, 2) = 3.0f; a.at(0, 3) = 4.0f;
  a.at(1, 0) = 5.0f; a.at(1, 1) = 6.0f; a.at(1, 2) = 7.0f; a.at(1, 3) = 8.0f;
  
  b.at(0, 0) = 9.0f; b.at(0, 1) = 10.0f; b.at(0, 2) = 11.0f; b.at(0, 3) = 12.0f;
  b.at(1, 0) = 13.0f; b.at(1, 1) = 14.0f; b.at(1, 2) = 15.0f; b.at(1, 3) = 16.0f;
  
  Matrix<float> c = Matrix<float>::add(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 4);
  EXPECT_FLOAT_EQ(c.at(0, 0), 10.0f);  // 1 + 9
  EXPECT_FLOAT_EQ(c.at(0, 1), 12.0f);  // 2 + 10
  EXPECT_FLOAT_EQ(c.at(0, 2), 14.0f);  // 3 + 11
  EXPECT_FLOAT_EQ(c.at(0, 3), 16.0f);  // 4 + 12
  EXPECT_FLOAT_EQ(c.at(1, 0), 18.0f);  // 5 + 13
  EXPECT_FLOAT_EQ(c.at(1, 1), 20.0f);  // 6 + 14
  EXPECT_FLOAT_EQ(c.at(1, 2), 22.0f);  // 7 + 15
  EXPECT_FLOAT_EQ(c.at(1, 3), 24.0f);  // 8 + 16
  
  Matrix<float>::setForceScalarAdd(false);
}

TEST(MatrixDoubleTest, AddScalar) {
  Matrix<double>::setForceScalarAdd(true);
  
  Matrix<double> a(2, 4);
  Matrix<double> b(2, 4);
  
  // Initialize matrices
  a.at(0, 0) = 1.0; a.at(0, 1) = 2.0; a.at(0, 2) = 3.0; a.at(0, 3) = 4.0;
  a.at(1, 0) = 5.0; a.at(1, 1) = 6.0; a.at(1, 2) = 7.0; a.at(1, 3) = 8.0;
  
  b.at(0, 0) = 9.0; b.at(0, 1) = 10.0; b.at(0, 2) = 11.0; b.at(0, 3) = 12.0;
  b.at(1, 0) = 13.0; b.at(1, 1) = 14.0; b.at(1, 2) = 15.0; b.at(1, 3) = 16.0;
  
  Matrix<double> c = Matrix<double>::add(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 4);
  EXPECT_DOUBLE_EQ(c.at(0, 0), 10.0);  // 1 + 9
  EXPECT_DOUBLE_EQ(c.at(0, 1), 12.0);  // 2 + 10
  EXPECT_DOUBLE_EQ(c.at(0, 2), 14.0);  // 3 + 11
  EXPECT_DOUBLE_EQ(c.at(0, 3), 16.0);  // 4 + 12
  EXPECT_DOUBLE_EQ(c.at(1, 0), 18.0);  // 5 + 13
  EXPECT_DOUBLE_EQ(c.at(1, 1), 20.0);  // 6 + 14
  EXPECT_DOUBLE_EQ(c.at(1, 2), 22.0);  // 7 + 15
  EXPECT_DOUBLE_EQ(c.at(1, 3), 24.0);  // 8 + 16
  
  Matrix<double>::setForceScalarAdd(false);
}

TEST(MatrixTest, AddScalarConsistency) {
  Matrix<float>::setForceScalarAdd(true);
  
  // Test that scalar and SIMD produce identical results
  Matrix<float> a(4, 4);
  Matrix<float> b(4, 4);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      a.at(i, j) = static_cast<float>(i * 4 + j + 1);
      b.at(i, j) = static_cast<float>((i * 4 + j + 1) * 2);
    }
  }
  
  Matrix<float> c_scalar = Matrix<float>::add(a, b);
  
  // Reset to allow SIMD
  Matrix<float>::setForceScalarAdd(false);
  Matrix<float> c_simd = Matrix<float>::add(a, b);
  
  // Compare results
  EXPECT_EQ(c_scalar.rows(), c_simd.rows());
  EXPECT_EQ(c_scalar.cols(), c_simd.cols());
  for (size_t i = 0; i < c_scalar.rows(); ++i) {
    for (size_t j = 0; j < c_scalar.cols(); ++j) {
      EXPECT_FLOAT_EQ(c_scalar.at(i, j), c_simd.at(i, j));
    }
  }
}

TEST(MatrixDoubleTest, AddScalarConsistency) {
  Matrix<double>::setForceScalarAdd(true);
  
  // Test that scalar and SIMD produce identical results
  Matrix<double> a(4, 4);
  Matrix<double> b(4, 4);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      a.at(i, j) = static_cast<double>(i * 4 + j + 1);
      b.at(i, j) = static_cast<double>((i * 4 + j + 1) * 2);
    }
  }
  
  Matrix<double> c_scalar = Matrix<double>::add(a, b);
  
  // Reset to allow SIMD
  Matrix<double>::setForceScalarAdd(false);
  Matrix<double> c_simd = Matrix<double>::add(a, b);
  
  // Compare results
  EXPECT_EQ(c_scalar.rows(), c_simd.rows());
  EXPECT_EQ(c_scalar.cols(), c_simd.cols());
  for (size_t i = 0; i < c_scalar.rows(); ++i) {
    for (size_t j = 0; j < c_scalar.cols(); ++j) {
      EXPECT_DOUBLE_EQ(c_scalar.at(i, j), c_simd.at(i, j));
    }
  }
}

// Tests for SSE4.2 addition (fallback SIMD path)
TEST(MatrixTest, AddSSE42) {
  Matrix<float>::setForceSSE42Add(true);
  
  Matrix<float> a(2, 4);  // Use multiple of 4 for better SSE testing
  Matrix<float> b(2, 4);
  
  // Initialize matrices
  a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f; a.at(0, 2) = 3.0f; a.at(0, 3) = 4.0f;
  a.at(1, 0) = 5.0f; a.at(1, 1) = 6.0f; a.at(1, 2) = 7.0f; a.at(1, 3) = 8.0f;
  
  b.at(0, 0) = 9.0f; b.at(0, 1) = 10.0f; b.at(0, 2) = 11.0f; b.at(0, 3) = 12.0f;
  b.at(1, 0) = 13.0f; b.at(1, 1) = 14.0f; b.at(1, 2) = 15.0f; b.at(1, 3) = 16.0f;
  
  Matrix<float> c = Matrix<float>::add(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 4);
  EXPECT_FLOAT_EQ(c.at(0, 0), 10.0f);  // 1 + 9
  EXPECT_FLOAT_EQ(c.at(0, 1), 12.0f);  // 2 + 10
  EXPECT_FLOAT_EQ(c.at(0, 2), 14.0f);  // 3 + 11
  EXPECT_FLOAT_EQ(c.at(0, 3), 16.0f);  // 4 + 12
  EXPECT_FLOAT_EQ(c.at(1, 0), 18.0f);  // 5 + 13
  EXPECT_FLOAT_EQ(c.at(1, 1), 20.0f);  // 6 + 14
  EXPECT_FLOAT_EQ(c.at(1, 2), 22.0f);  // 7 + 15
  EXPECT_FLOAT_EQ(c.at(1, 3), 24.0f);  // 8 + 16
  
  Matrix<float>::setForceSSE42Add(false);
}

TEST(MatrixDoubleTest, AddSSE42) {
  Matrix<double>::setForceSSE42Add(true);
  
  Matrix<double> a(2, 4);  // Use multiple of 4 for better SSE testing
  Matrix<double> b(2, 4);
  
  // Initialize matrices
  a.at(0, 0) = 1.0; a.at(0, 1) = 2.0; a.at(0, 2) = 3.0; a.at(0, 3) = 4.0;
  a.at(1, 0) = 5.0; a.at(1, 1) = 6.0; a.at(1, 2) = 7.0; a.at(1, 3) = 8.0;
  
  b.at(0, 0) = 9.0; b.at(0, 1) = 10.0; b.at(0, 2) = 11.0; b.at(0, 3) = 12.0;
  b.at(1, 0) = 13.0; b.at(1, 1) = 14.0; b.at(1, 2) = 15.0; b.at(1, 3) = 16.0;
  
  Matrix<double> c = Matrix<double>::add(a, b);
  
  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 4);
  EXPECT_DOUBLE_EQ(c.at(0, 0), 10.0);  // 1 + 9
  EXPECT_DOUBLE_EQ(c.at(0, 1), 12.0);  // 2 + 10
  EXPECT_DOUBLE_EQ(c.at(0, 2), 14.0);  // 3 + 11
  EXPECT_DOUBLE_EQ(c.at(0, 3), 16.0);  // 4 + 12
  EXPECT_DOUBLE_EQ(c.at(1, 0), 18.0);  // 5 + 13
  EXPECT_DOUBLE_EQ(c.at(1, 1), 20.0);  // 6 + 14
  EXPECT_DOUBLE_EQ(c.at(1, 2), 22.0);  // 7 + 15
  EXPECT_DOUBLE_EQ(c.at(1, 3), 24.0);  // 8 + 16
  
  Matrix<double>::setForceSSE42Add(false);
}

TEST(MatrixTest, AddSSE42Consistency) {
  Matrix<float>::setForceSSE42Add(true);
  
  // Test that SSE4.2 and AVX-512 produce identical results (if AVX-512 available)
  Matrix<float> a(4, 8);  // Use dimensions that work well with both
  Matrix<float> b(4, 8);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      a.at(i, j) = static_cast<float>(i * 8 + j + 1);
      b.at(i, j) = static_cast<float>((i * 8 + j + 1) * 2);
    }
  }
  
  Matrix<float> c_sse42 = Matrix<float>::add(a, b);
  
  // Reset to allow AVX-512
  Matrix<float>::setForceSSE42Add(false);
  Matrix<float> c_avx512 = Matrix<float>::add(a, b);
  
  // Compare results
  EXPECT_EQ(c_sse42.rows(), c_avx512.rows());
  EXPECT_EQ(c_sse42.cols(), c_avx512.cols());
  for (size_t i = 0; i < c_sse42.rows(); ++i) {
    for (size_t j = 0; j < c_sse42.cols(); ++j) {
      EXPECT_FLOAT_EQ(c_sse42.at(i, j), c_avx512.at(i, j));
    }
  }
}

TEST(MatrixDoubleTest, AddSSE42Consistency) {
  Matrix<double>::setForceSSE42Add(true);
  
  // Test that SSE4.2 and AVX-512 produce identical results (if AVX-512 available)
  Matrix<double> a(4, 8);  // Use dimensions that work well with both
  Matrix<double> b(4, 8);
  
  // Fill with test data
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      a.at(i, j) = static_cast<double>(i * 8 + j + 1);
      b.at(i, j) = static_cast<double>((i * 8 + j + 1) * 2);
    }
  }
  
  Matrix<double> c_sse42 = Matrix<double>::add(a, b);
  
  // Reset to allow AVX-512
  Matrix<double>::setForceSSE42Add(false);
  Matrix<double> c_avx512 = Matrix<double>::add(a, b);
  
  // Compare results
  EXPECT_EQ(c_sse42.rows(), c_avx512.rows());
  EXPECT_EQ(c_sse42.cols(), c_avx512.cols());
  for (size_t i = 0; i < c_sse42.rows(); ++i) {
    for (size_t j = 0; j < c_sse42.cols(); ++j) {
      EXPECT_DOUBLE_EQ(c_sse42.at(i, j), c_avx512.at(i, j));
    }
  }
}

// Tests for subtraction (minus)
TEST(MatrixTest, SubBasic) {
  Matrix<float> a({{5, 6}, {7, 8}});
  Matrix<float> b({{1, 2}, {3, 4}});
  Matrix<float> c = Matrix<float>::sub(a, b);

  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_NEAR(c.at(0, 0), 4.0f, 1e-5f);  // 5 - 1
  EXPECT_NEAR(c.at(0, 1), 4.0f, 1e-5f);  // 6 - 2
  EXPECT_NEAR(c.at(1, 0), 4.0f, 1e-5f);  // 7 - 3
  EXPECT_NEAR(c.at(1, 1), 4.0f, 1e-5f);  // 8 - 4
}

TEST(MatrixDoubleTest, SubBasic) {
  Matrix<double> a({{5, 6}, {7, 8}});
  Matrix<double> b({{1, 2}, {3, 4}});
  Matrix<double> c = Matrix<double>::sub(a, b);

  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 2);
  EXPECT_NEAR(c.at(0, 0), 4.0, 1e-10);
  EXPECT_NEAR(c.at(0, 1), 4.0, 1e-10);
  EXPECT_NEAR(c.at(1, 0), 4.0, 1e-10);
  EXPECT_NEAR(c.at(1, 1), 4.0, 1e-10);
}

TEST(MatrixTest, SubDimensionMismatch) {
  Matrix<float> a({{1, 2}, {3, 4}});
  Matrix<float> b({{1, 2}, {3, 4}, {5, 6}});
  EXPECT_THROW(Matrix<float>::sub(a, b), std::invalid_argument);
}

TEST(MatrixTest, SubWithNegatives) {
  Matrix<float> a({{-1, -2}, {3, -4}});
  Matrix<float> b({{1, -2}, {-3, 4}});
  Matrix<float> c = a - b; // operator-

  EXPECT_FLOAT_EQ(c.at(0,0), -2.0f); // -1 - 1
  EXPECT_FLOAT_EQ(c.at(0,1), 0.0f);  // -2 - (-2)
  EXPECT_FLOAT_EQ(c.at(1,0), 6.0f);  // 3 - (-3)
  EXPECT_FLOAT_EQ(c.at(1,1), -8.0f); // -4 - 4
}

TEST(MatrixTest, SubScalarAndSSEConsistency) {
  // Ensure scalar and SIMD subtraction paths match
  Matrix<float> a(4,4);
  Matrix<float> b(4,4);
  for (size_t i=0;i<4;i++) for (size_t j=0;j<4;j++) {
    a.at(i,j) = static_cast<float>(i*4 + j + 1);
    b.at(i,j) = static_cast<float>((i*4 + j + 1) * 2);
  }

  Matrix<float>::setForceScalarSub(true);
  Matrix<float> c_scalar = Matrix<float>::sub(a,b);
  Matrix<float>::setForceScalarSub(false);

  Matrix<float>::setForceSSE42Sub(true);
  Matrix<float> c_sse = Matrix<float>::sub(a,b);
  Matrix<float>::setForceSSE42Sub(false);

  for (size_t i=0;i<c_scalar.rows();++i) for (size_t j=0;j<c_scalar.cols();++j)
    EXPECT_FLOAT_EQ(c_scalar.at(i,j), c_sse.at(i,j));
}

TEST(MatrixTest, ChainedMultiplyAddFloat) {
  Matrix<float> c({{1,2,3},{4,5,6}}); // 2x3
  Matrix<float> b({{1,0},{0,1},{1,0}}); // 3x2
  Matrix<float> d({{1,1},{1,1}}); // 2x2

  Matrix<float> a = (c * b) + d;

  EXPECT_EQ(a.rows(), 2);
  EXPECT_EQ(a.cols(), 2);
  EXPECT_FLOAT_EQ(a.at(0,0), 5.0f);  // 4 + 1
  EXPECT_FLOAT_EQ(a.at(0,1), 3.0f);  // 2 + 1
  EXPECT_FLOAT_EQ(a.at(1,0), 11.0f); // 10 + 1
  EXPECT_FLOAT_EQ(a.at(1,1), 6.0f);  // 5 + 1
}

TEST(MatrixDoubleTest, ChainedMultiplyAddDouble) {
  Matrix<double> c({{1,2,3},{4,5,6}}); // 2x3
  Matrix<double> b({{1,0},{0,1},{1,0}}); // 3x2
  Matrix<double> d({{1,1},{1,1}}); // 2x2

  Matrix<double> a = (c * b) + d;

  EXPECT_EQ(a.rows(), 2);
  EXPECT_EQ(a.cols(), 2);
  EXPECT_DOUBLE_EQ(a.at(0,0), 5.0);  // 4 + 1
  EXPECT_DOUBLE_EQ(a.at(0,1), 3.0);  // 2 + 1
  EXPECT_DOUBLE_EQ(a.at(1,0), 11.0); // 10 + 1
  EXPECT_DOUBLE_EQ(a.at(1,1), 6.0);  // 5 + 1
}
