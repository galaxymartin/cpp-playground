#include "matrix.hpp"
#include <iostream>

int main() {
    Matrix<float> a({{1, 2}, {3, 4}});
    Matrix<float> b({{5, 6}, {7, 8}});
    
    // Test operator+
    Matrix<float> c = a + b;
    std::cout << "a + b:" << std::endl;
    for (size_t i = 0; i < c.rows(); ++i) {
        for (size_t j = 0; j < c.cols(); ++j) {
            std::cout << c.at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    // Test operator*
    Matrix<float> d = a * b;
    std::cout << "a * b:" << std::endl;
    for (size_t i = 0; i < d.rows(); ++i) {
        for (size_t j = 0; j < d.cols(); ++j) {
            std::cout << d.at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}