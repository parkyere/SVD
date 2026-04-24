//
// SVD 라이브러리 demo — svd.hpp 공개 API만 사용.
// (Hestenes Jacobi, Householder, QRCP 등 내부 알고리즘은 detail/* 에 캡슐화되어 보이지 않음)
//

#include "svd.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>

namespace {

void print_matrix(const svd::Matrix& M, const std::string& name) {
    constexpr std::size_t MAX_PRINT_DIM = 16;
    std::cout << "=== " << name << " (" << M.rows() << "x" << M.cols() << ") ===\n";
    if (M.rows() > MAX_PRINT_DIM || M.cols() > MAX_PRINT_DIM) {
        std::cout << "(too large to print: limit "
            << MAX_PRINT_DIM << "x" << MAX_PRINT_DIM << ")\n\n";
        return;
    }
    for (std::size_t i = 0; i < M.rows(); ++i) {
        for (std::size_t j = 0; j < M.cols(); ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << M(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void print_dims(const svd::SVDResult& r) {
    std::cout << "U dim: " << r.U.rows() << "x" << r.U.cols()
        << ", V dim: " << r.V.rows() << "x" << r.V.cols() << "\n\n";
}

} // anonymous namespace

int main() {
    try {
        svd::Matrix A = svd::Matrix::from_row_major(5, 3, {
            1.0,  5.0,  9.0,
            2.0,  6.0, 10.0,
            3.0,  7.0, 11.0,
            4.0,  8.0, 12.0,
            5.0,  0.0,  2.0
            });

        std::cout << "[Test 1: M >= N Matrix, Thin SVD]\n";
        auto resA = svd::compute_svd(A);
        print_matrix(resA.Sigma, "Matrix Sigma (Thin)");
        print_dims(resA);
        print_matrix(svd::matmul(svd::matmul(resA.U, resA.Sigma), resA.V.transpose()),
            "Reconstructed A");

        svd::Matrix B = A.transpose();
        std::cout << "[Test 2: M < N Matrix, Thin SVD]\n";
        auto resB = svd::compute_svd(B);
        print_matrix(resB.Sigma, "Matrix Sigma (Thin)");
        print_dims(resB);
        print_matrix(svd::matmul(svd::matmul(resB.U, resB.Sigma), resB.V.transpose()),
            "Reconstructed B");

        std::cout << "[Test 3: M >= N Matrix, Full SVD — U는 5x5 정사각]\n";
        auto resA_full = svd::compute_svd(A, svd::SVDMode::Full);
        print_matrix(resA_full.Sigma, "Matrix Sigma (Full, 5x3)");
        print_dims(resA_full);
        print_matrix(svd::matmul(svd::matmul(resA_full.U, resA_full.Sigma),
            resA_full.V.transpose()), "Reconstructed A (Full)");

        std::cout << "[Test 4: M < N Matrix, Full SVD — V는 5x5 정사각]\n";
        auto resB_full = svd::compute_svd(B, svd::SVDMode::Full);
        print_matrix(resB_full.Sigma, "Matrix Sigma (Full, 3x5)");
        print_dims(resB_full);
        print_matrix(svd::matmul(svd::matmul(resB_full.U, resB_full.Sigma),
            resB_full.V.transpose()), "Reconstructed B (Full)");

        // [Test 5: OOM 방어 가드 — 주석 해제 시 length_error로 안전하게 거부됨]
        // svd::Matrix malicious(2000000, 2000000); // 약 32 TB 요청

    }
    catch (const std::length_error& e) {
        std::cerr << "[SAFE REJECT] Computation Refused: " << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
    }
    return 0;
}
