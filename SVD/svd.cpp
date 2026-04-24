//
// SVD 공개 API 구현.
// 내부 알고리즘 (Jacobi, QRCP, Householder)은 detail/* 헤더에 캡슐화.
//

#include "svd.hpp"
#include "detail/allocator.hpp"
#include "detail/svd_impl.hpp"
#include "detail/jacobi.hpp"

#include <cassert>
#include <cstring>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <execution>
#include <ranges>
#include <numeric>
#include <cmath>

namespace svd {

// =======================================================================
// Matrix
// =======================================================================

std::size_t Matrix::calculate_stride(std::size_t r) {
    if (r == 0) return 0;
    if (r > std::numeric_limits<std::size_t>::max() - 15)
        throw std::length_error("row dimension too large for stride padding");
    std::size_t s = (r + 7) & ~std::size_t{ 7 };
    if (s >= 64 && (s & (s - 1)) == 0) s += 8; // 2의 거듭제곱은 bank conflict 회피용 +1 cache line
    return s;
}

Matrix::Matrix() noexcept
    : rows_(0), cols_(0), stride_(0), data_(nullptr) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double val)
    : rows_(rows), cols_(cols), stride_(calculate_stride(rows)) {
    const std::size_t total = detail::safe_mul(stride_, cols);
    if (total > 0) {
        data_ = detail::allocate_aligned_uninitialized(total);
        if (val != 0.0) {
            double* ptr = std::assume_aligned<64>(data_.get());
            std::fill_n(std::execution::par_unseq, ptr, total, val);
        } else {
            std::memset(data_.get(), 0, detail::safe_mul(total, sizeof(double)));
        }
    }
}

Matrix::Matrix(std::size_t rows, std::size_t cols, UninitializedTag)
    : rows_(rows), cols_(cols), stride_(calculate_stride(rows)) {
    const std::size_t total = detail::safe_mul(stride_, cols);
    if (total > 0) data_ = detail::allocate_aligned_uninitialized(total);
}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_) {
    const std::size_t total = detail::safe_mul(stride_, cols_);
    if (total > 0) {
        data_ = detail::allocate_aligned_uninitialized(total);
        const double* src = std::assume_aligned<64>(other.data_.get());
        double* dst = std::assume_aligned<64>(data_.get());
        std::copy(std::execution::par_unseq, src, src + total, dst);
    }
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(std::exchange(other.rows_, 0)),
      cols_(std::exchange(other.cols_, 0)),
      stride_(std::exchange(other.stride_, 0)),
      data_(std::move(other.data_)) {}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    rows_ = std::exchange(other.rows_, 0);
    cols_ = std::exchange(other.cols_, 0);
    stride_ = std::exchange(other.stride_, 0);
    data_ = std::move(other.data_);
    return *this;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        Matrix tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

Matrix::~Matrix() = default;

double& Matrix::operator()(std::size_t i, std::size_t j) noexcept {
    assert(i < rows_ && j < cols_ && "Matrix::operator() index out of range");
    return data_.get()[j * stride_ + i];
}

const double& Matrix::operator()(std::size_t i, std::size_t j) const noexcept {
    assert(i < rows_ && j < cols_ && "Matrix::operator() index out of range");
    return data_.get()[j * stride_ + i];
}

double& Matrix::at(std::size_t i, std::size_t j) {
    if (i >= rows_ || j >= cols_)
        throw std::out_of_range("Matrix::at index out of range");
    return data_.get()[j * stride_ + i];
}

const double& Matrix::at(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_)
        throw std::out_of_range("Matrix::at index out of range");
    return data_.get()[j * stride_ + i];
}

Matrix Matrix::transpose() const {
    Matrix T(cols_, rows_, UninitializedTag{});
    constexpr std::size_t BLOCK_SIZE = 32;
    auto block_j_indices = std::views::iota(std::size_t{ 0 }, (cols_ + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const bool use_par = (rows_ * cols_ >= 10000);

    auto kernel = [&](std::size_t bj) noexcept {
        const std::size_t j_start = bj * BLOCK_SIZE;
        const std::size_t j_end = std::min(j_start + BLOCK_SIZE, cols_);
        for (std::size_t bi = 0; bi < rows_; bi += BLOCK_SIZE) {
            const std::size_t i_end = std::min(bi + BLOCK_SIZE, rows_);
            for (std::size_t j = j_start; j < j_end; ++j) {
                for (std::size_t i = bi; i < i_end; ++i) T(j, i) = (*this)(i, j);
            }
        }
    };

    if (use_par)
        std::for_each(std::execution::par_unseq, block_j_indices.begin(), block_j_indices.end(), kernel);
    else
        std::for_each(std::execution::seq, block_j_indices.begin(), block_j_indices.end(), kernel);

    return T;
}

Matrix Matrix::from_row_major(std::size_t r, std::size_t c, const std::vector<double>& flat) {
    const std::size_t needed = detail::safe_mul(r, c);
    if (flat.size() < needed)
        throw std::invalid_argument("flat vector size insufficient for matrix dimensions");
    Matrix M(r, c, 0.0);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            M(i, j) = flat[i * c + j];
    return M;
}

// =======================================================================
// SVDResult
// =======================================================================

SVDResult::SVDResult(Matrix&& u, Matrix&& s, Matrix&& v) noexcept
    : U(std::move(u)), Sigma(std::move(s)), V(std::move(v)) {}

// =======================================================================
// compute_svd — Thin / Full
// =======================================================================

SVDResult compute_svd(const Matrix& A, SVDMode mode) {
    if (A.rows() == 0 || A.cols() == 0)
        return SVDResult(Matrix(A), Matrix(A), Matrix(A));

    // NaN/Inf 검증 — 열 단위로 par_unseq 병렬
    auto col_indices = std::views::iota(std::size_t{ 0 }, A.cols());
    const bool has_invalid = std::any_of(std::execution::par_unseq,
        col_indices.begin(), col_indices.end(),
        [&](std::size_t j) noexcept {
            const double* col = detail::MatrixAccess::data(A) + j * detail::MatrixAccess::stride(A);
            return std::any_of(std::execution::unseq, col, col + A.rows(),
                [](double x) noexcept { return !std::isfinite(x); });
        });
    if (has_invalid)
        throw std::invalid_argument("Input matrix contains NaN or Inf");

    const std::size_t M = A.rows(), N = A.cols();

    // 1단계: thin SVD (2단계 QRCP 전처리 + Jacobi)
    SVDResult thin = (M >= N)
        ? detail::thin_svd_with_qrcp(A)
        : [&]() {
            Matrix At = A.transpose();
            SVDResult rt = detail::thin_svd_with_qrcp(At);
            // A = (Aᵀ)ᵀ = (Uₜ Σₜ Vₜᵀ)ᵀ = Vₜ Σₜ Uₜᵀ  →  U=Vₜ, Σ=Σₜ, V=Uₜ
            return SVDResult(std::move(rt.V), std::move(rt.Sigma), std::move(rt.U));
        }();

    if (mode == SVDMode::Thin) return thin;

    // 2단계: Full 모드 — 누락된 정사각 직교 보강
    if (M > N) {
        Matrix U_full = detail::extend_to_orthogonal(thin.U);
        Matrix Sigma_full = detail::make_padded_sigma(M, N, thin.Sigma);
        return SVDResult(std::move(U_full), std::move(Sigma_full), std::move(thin.V));
    } else if (M < N) {
        Matrix V_full = detail::extend_to_orthogonal(thin.V);
        Matrix Sigma_full = detail::make_padded_sigma(M, N, thin.Sigma);
        return SVDResult(std::move(thin.U), std::move(Sigma_full), std::move(V_full));
    }
    return thin; // M == N
}

// =======================================================================
// matmul — 검증·시연 유틸리티
// =======================================================================

Matrix matmul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows(), B.cols(), Matrix::UninitializedTag{});
    auto I_indices = std::views::iota(std::size_t{ 0 }, A.rows());
    auto J_indices = std::views::iota(std::size_t{ 0 }, B.cols());
    auto K_indices = std::views::iota(std::size_t{ 0 }, A.cols());

    std::for_each(std::execution::par_unseq, J_indices.begin(), J_indices.end(),
        [&](std::size_t j) noexcept {
            std::for_each(std::execution::unseq, I_indices.begin(), I_indices.end(),
                [&](std::size_t i) noexcept {
                    C(i, j) = std::transform_reduce(std::execution::unseq,
                        K_indices.begin(), K_indices.end(), 0.0, std::plus<>{},
                        [&](std::size_t k) noexcept { return A(i, k) * B(k, j); });
                });
        });
    return C;
}

} // namespace svd
