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
//

#include <cstddef>
#include <vector>
#include <memory>
#include <new>      // std::align_val_t
#include <cassert>
#include <limits>
#include <stdexcept>

#ifndef SVD_MAX_MATRIX_BYTES
#define SVD_MAX_MATRIX_BYTES (8ULL << 30)  // 행렬 1개의 안전 상한 (default 8 GiB)
#endif

namespace svd {

namespace detail {
    // unique_ptr deleter — _Compressed_pair의 EBO 검사가 complete type을 요구하므로 inline 정의.
    struct AlignedFree {
        inline void operator()(void* p) const noexcept {
            ::operator delete(p, std::align_val_t{ 64 });
        }
    };
    using AlignedDoublePtr = std::unique_ptr<double[], AlignedFree>;
}

class SVD;  // 전방 선언 — Matrix가 friend로 지정

// =======================================================================
// Matrix — 64-byte 정렬 column-major 밀집 행렬.
// 내부적으로 cache-line padding된 stride 사용 (LAPACK의 'lda').
// =======================================================================
class Matrix {
public:
    struct UninitializedTag {};

    Matrix() noexcept;
    Matrix(std::size_t rows, std::size_t cols, double val = 0.0);
    Matrix(std::size_t rows, std::size_t cols, UninitializedTag);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix();

    constexpr std::size_t rows() const noexcept { return rows_; }
    constexpr std::size_t cols() const noexcept { return cols_; }

    inline double& operator()(std::size_t i, std::size_t j) noexcept {
        assert(i < rows_ && j < cols_ && "Matrix::operator() index out of range");
        return data_.get()[j * stride_ + i];
    }
    inline const double& operator()(std::size_t i, std::size_t j) const noexcept {
        assert(i < rows_ && j < cols_ && "Matrix::operator() index out of range");
        return data_.get()[j * stride_ + i];
    }
    double& at(std::size_t i, std::size_t j);
    const double& at(std::size_t i, std::size_t j) const;

    Matrix transpose() const;

    static Matrix from_row_major(std::size_t r, std::size_t c, const std::vector<double>& flat);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t stride_;
    detail::AlignedDoublePtr data_;

    static constexpr std::size_t calculate_stride(std::size_t r) {
        if (r == 0) return 0;
        if (r > std::numeric_limits<std::size_t>::max() - 15)
            throw std::length_error("row dimension too large for stride padding");
        std::size_t s = (r + 7) & ~std::size_t{ 7 };
        if (s >= 64 && (s & (s - 1)) == 0) s += 8; // 2의 거듭제곱은 bank conflict 회피용 +1 cache line
        return s;
    }

    // 내부 SVD 알고리즘이 SIMD 정렬 raw pointer를 직접 다뤄야 하므로 friend.
    friend class SVD;
};

// =======================================================================
// SVD 결과 — A = U · Σ · Vᵀ
// =======================================================================
struct SVDResult {
    Matrix U, Sigma, V;
    SVDResult(Matrix&& u, Matrix&& s, Matrix&& v) noexcept;
};

enum class SVDMode {
    Thin,   // 메모리 절약: U 또는 V 한쪽이 직사각 (정규직교 열)
    Full    // 정사각 직교 U·V; Σ는 (rows × cols) 패딩 대각
};

// =======================================================================
// SVD — 하나의 SVD 계산 작업을 캡슐화한 클래스.
// 입력 검증, 두 단계 QRCP 전처리, Hestenes Jacobi, 결과 합성, Full 보강을
// 단계별 멤버로 보유한다. 모든 알고리즘 내부 구조(Householder, QRCP primitives,
// 중간 행렬 W1/W2/U_R2/V_R2 등)는 private — 외부에서 접근·관찰 불가.
// =======================================================================
class SVD {
public:
    explicit SVD(const Matrix& A) noexcept;
    SVDResult compute(SVDMode mode = SVDMode::Thin);

private:
    // ─── 파이프라인 단계별 누적 상태 ──────────────────────────────────
    const Matrix& input_;
    Matrix work_;                       // input_ 또는 input_.transpose() (rows ≥ cols 보장)
    bool transposed_ = false;

    Matrix W1_;                         // QRCP stage 1: implicit Q1 + R1 (upper triangle)
    std::vector<double> taus1_;
    std::vector<std::size_t> perm1_;

    Matrix W2_;                         // QRCP stage 2: implicit Q2 + R2 (upper triangle)
    std::vector<double> taus2_;

    Matrix U_R2_, V_R2_, Sigma_;        // SVD of R2

    // ─── 파이프라인 단계 메서드 (state 사용) ─────────────────────────
    void validate_finite_();
    void normalize_orientation_();
    void preprocess_qrcp_stage1_();
    void preprocess_qrcp_stage2_();
    void jacobi_on_R2_();
    SVDResult compose_thin_();
    SVDResult promote_to_full_(SVDResult thin);

    // ─── Static building blocks (state 없음, SVD 외부에서 호출 불가) ─
    struct Householder {
        std::vector<double> v;          // length n; v[0] == 1
        double tau;                     // 2 / (vᵀv)
        double alpha;                   // 반사 후 spike 값
    };
    static Householder make_householder_(const double* x, std::size_t n);
    static void apply_householder_(const Householder& h, double* y) noexcept;

    static void businger_golub_inplace_(Matrix& W, std::vector<double>& taus,
                                         std::vector<std::size_t>& perm);
    static void qr_no_pivot_inplace_(Matrix& W, std::vector<double>& taus);

    static Matrix extract_R_(const Matrix& W);
    static void apply_Q_left_inplace_(const Matrix& W, const std::vector<double>& taus,
                                        Matrix& target);
    static Matrix apply_Q_to_thin_(const Matrix& W, const std::vector<double>& taus,
                                     const Matrix& U_R);
    static Matrix permute_rows_square_(const std::vector<std::size_t>& perm,
                                         const Matrix& V_in);
    static Matrix extend_to_orthogonal_(const Matrix& Q);
    static Matrix make_padded_sigma_(std::size_t target_M, std::size_t target_N,
                                       const Matrix& Sigma_diag);

    // Hestenes one-sided Jacobi engine — R2 (작고 잘 조건화된 N×N) 위에서 실행.
    static SVDResult jacobi_engine_(const Matrix& A);
};

// 편의 wrapper — 한 줄로 SVD 계산.
inline SVDResult compute_svd(const Matrix& A, SVDMode mode = SVDMode::Thin) {
    return SVD(A).compute(mode);
}

// 행렬 곱 — 검증·시연 유틸리티.
Matrix matmul(const Matrix& A, const Matrix& B);

} // namespace svd
