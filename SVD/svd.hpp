#pragma once
//
// SVD — Hestenes one-sided Jacobi 기반 CPU SVD 라이브러리 (학습용).
// 사용자는 이 헤더 하나만 include하면 된다.
//
// 사용 예:
//     #include "svd.hpp"
//     svd::Matrix A = svd::Matrix::from_row_major(M, N, flat_data);
//     svd::SVDResult r = svd::compute_svd(A);                     // Thin
//     svd::SVDResult r = svd::compute_svd(A, svd::SVDMode::Full); // Full
//     // r.U, r.Sigma, r.V — 모두 svd::Matrix
//

#include <cstddef>
#include <vector>
#include <memory>
#include <new>      // std::align_val_t

// 컴파일 시 -DSVD_MAX_MATRIX_BYTES=... 로 행렬 1개의 안전 상한 재정의 가능 (default 8 GiB).
#ifndef SVD_MAX_MATRIX_BYTES
#define SVD_MAX_MATRIX_BYTES (8ULL << 30)
#endif

namespace svd {

namespace detail {
    // AlignedFree는 unique_ptr의 deleter — 빈 struct이므로 EBO를 위해 complete type이어야 함.
    // (compressed_pair<Deleter, T*>의 EBO 검사가 __is_empty<Deleter>를 요구)
    struct AlignedFree {
        void operator()(void* p) const noexcept {
            ::operator delete(p, std::align_val_t{ 64 });
        }
    };
    using AlignedDoublePtr = std::unique_ptr<double[], AlignedFree>;

    // 사용자에게 노출되는 다른 detail 이름들은 forward declaration만.
    class MatrixAccess;
}

// =======================================================================
// Matrix — 64-byte 정렬 column-major 밀집 행렬.
// 내부적으로 cache-line padding된 stride를 사용 (LAPACK의 'lda'와 동일 개념).
// =======================================================================
class Matrix {
public:
    // 0-초기화 회피용 태그 — performance critical 경로에서만 사용.
    struct UninitializedTag {};

    Matrix() noexcept;
    Matrix(std::size_t rows, std::size_t cols, double val = 0.0);
    Matrix(std::size_t rows, std::size_t cols, UninitializedTag);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix();

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    // Hot-path 접근자 — debug에서만 bounds check (NDEBUG에서 무비용).
    double& operator()(std::size_t i, std::size_t j) noexcept;
    const double& operator()(std::size_t i, std::size_t j) const noexcept;

    // Release에서도 bounds check (외부 입력 등 안전이 필요한 곳).
    double& at(std::size_t i, std::size_t j);
    const double& at(std::size_t i, std::size_t j) const;

    Matrix transpose() const;

    // Row-major flat vector → column-major Matrix 변환.
    static Matrix from_row_major(std::size_t r, std::size_t c, const std::vector<double>& flat);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t stride_;  // 메모리상 행 길이 (cache padding 포함)
    detail::AlignedDoublePtr data_;

    static std::size_t calculate_stride(std::size_t r);

    // 내부 SVD 알고리즘은 raw 포인터·stride를 직접 다뤄야 SIMD를 살릴 수 있음.
    // detail::MatrixAccess가 private 멤버 접근의 유일한 통로 — 사용자 영역에서는 정의 미공개.
    friend class detail::MatrixAccess;
};

// =======================================================================
// SVD 결과 — A = U · Σ · Vᵀ
// =======================================================================
struct SVDResult {
    Matrix U, Sigma, V;
    SVDResult(Matrix&& u, Matrix&& s, Matrix&& v) noexcept;
};

enum class SVDMode {
    Thin,   // 메모리 절약 — U 또는 V 한쪽이 직사각 (정규직교 열)
    Full    // 정사각 직교 U, V; Σ는 (rows × cols) 패딩 대각
};

// 입력 행렬 A의 SVD 계산.
// - 항상 A = U · Σ · Vᵀ 만족 (수치 오차 한도 내).
// - NaN / Inf 입력 시 std::invalid_argument throw.
// - 메모리 안전 한계 초과 시 std::length_error throw.
SVDResult compute_svd(const Matrix& A, SVDMode mode = SVDMode::Thin);

// 행렬 곱 — 검증·시연 유틸리티 (재구성 검사 등에 사용).
Matrix matmul(const Matrix& A, const Matrix& B);

} // namespace svd
