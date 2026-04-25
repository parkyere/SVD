//
// SVD 라이브러리 구현 — Matrix, SVD 클래스, matmul 한 파일.
// SVD class의 private 메서드들이 이 파일 안에서 모두 정의된다.
//

#include "svd.hpp"

#include <cassert>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <execution>
#include <ranges>
#include <numeric>
#include <thread>
#include <span>
#include <memory>

namespace svd {

// =======================================================================
// [1] 내부 allocator helpers — TU-private (anonymous namespace)
// =======================================================================
namespace {

inline constexpr std::size_t MAX_MATRIX_DOUBLES = SVD_MAX_MATRIX_BYTES / sizeof(double);

constexpr std::size_t safe_mul(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("matrix dimension multiplication overflow");
    return a * b;
}

void check_element_count(std::size_t count) {
    if (count > MAX_MATRIX_DOUBLES) {
        throw std::length_error(
            "matrix allocation exceeds safety limit ("
            + std::to_string(SVD_MAX_MATRIX_BYTES >> 30) + " GiB; requested "
            + std::to_string(count * sizeof(double)) + " bytes)");
    }
}

detail::AlignedDoublePtr allocate_aligned_uninitialized(std::size_t count) {
    if (count == 0) return nullptr;
    check_element_count(count);                                      // 1차: 안전 상한
    const std::size_t bytes = safe_mul(count, sizeof(double));       // 2차: overflow
    const std::size_t alloc_bytes = (bytes + 63) & ~std::size_t{ 63 };
    if (alloc_bytes < bytes)                                         // 3차: 반올림 wrap
        throw std::length_error("matrix allocation byte rounding overflow");
    void* ptr = ::operator new(alloc_bytes, std::align_val_t{ 64 });
    return detail::AlignedDoublePtr(static_cast<double*>(ptr));
}

// PQR fused reduction — Hestenes Jacobi의 column pair에 대해 한 pass로
// (vᵢᵀvⱼ, ||vᵢ||², ||vⱼ||²) 를 동시 누적.
struct PQR { double p, q, r; };
inline constexpr auto pqr_combine = [](const PQR& a, const PQR& b) noexcept -> PQR {
    return { a.p + b.p, a.q + b.q, a.r + b.r };
};
inline constexpr auto pqr_map = [](double vi, double vj) noexcept -> PQR {
    return { vi * vj, vi * vi, vj * vj };
};

}  // anonymous namespace

// =======================================================================
// [2] Matrix
// =======================================================================

Matrix::Matrix() noexcept : rows_(0), cols_(0), stride_(0), data_(nullptr) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double val)
    : rows_(rows), cols_(cols), stride_(calculate_stride(rows)) {
    const std::size_t total = safe_mul(stride_, cols);
    if (total > 0) {
        data_ = allocate_aligned_uninitialized(total);
        if (val != 0.0) {
            double* ptr = std::assume_aligned<64>(data_.get());
            std::fill_n(std::execution::par_unseq, ptr, total, val);
        } else {
            std::memset(data_.get(), 0, safe_mul(total, sizeof(double)));
        }
    }
}

Matrix::Matrix(std::size_t rows, std::size_t cols, UninitializedTag)
    : rows_(rows), cols_(cols), stride_(calculate_stride(rows)) {
    const std::size_t total = safe_mul(stride_, cols);
    if (total > 0) data_ = allocate_aligned_uninitialized(total);
}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_) {
    const std::size_t total = safe_mul(stride_, cols_);
    if (total > 0) {
        data_ = allocate_aligned_uninitialized(total);
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
    const std::size_t needed = safe_mul(r, c);
    if (flat.size() < needed)
        throw std::invalid_argument("flat vector size insufficient for matrix dimensions");
    Matrix M(r, c, 0.0);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            M(i, j) = flat[i * c + j];
    return M;
}

// =======================================================================
// [3] SVDResult
// =======================================================================

SVDResult::SVDResult(Matrix&& u, Matrix&& s, Matrix&& v) noexcept
    : U(std::move(u)), Sigma(std::move(s)), V(std::move(v)) {}

// =======================================================================
// [4] SVD — 파이프라인 구성·실행
// =======================================================================

SVD::SVD(const Matrix& A) noexcept : input_(A) {}

SVDResult SVD::compute(SVDMode mode) {
    if (input_.rows() == 0 || input_.cols() == 0)
        return SVDResult(Matrix(input_), Matrix(input_), Matrix(input_));

    validate_finite_();
    normalize_orientation_();

    // 자명 case: N <= 1이면 QRCP 무의미 — Jacobi engine으로 직접
    if (work_.cols() <= 1) {
        SVDResult thin = jacobi_engine_(work_);
        if (transposed_)
            thin = SVDResult(std::move(thin.V), std::move(thin.Sigma), std::move(thin.U));
        return mode == SVDMode::Full ? promote_to_full_(std::move(thin)) : thin;
    }

    preprocess_qrcp_stage1_();
    preprocess_qrcp_stage2_();
    jacobi_on_R2_();
    SVDResult thin = compose_thin_();

    if (transposed_)
        thin = SVDResult(std::move(thin.V), std::move(thin.Sigma), std::move(thin.U));

    return mode == SVDMode::Full ? promote_to_full_(std::move(thin)) : thin;
}

void SVD::validate_finite_() {
    auto col_indices = std::views::iota(std::size_t{ 0 }, input_.cols());
    const bool has_invalid = std::any_of(std::execution::par_unseq,
        col_indices.begin(), col_indices.end(),
        [&](std::size_t j) noexcept {
            const double* col = input_.data_.get() + j * input_.stride_;
            return std::any_of(std::execution::unseq, col, col + input_.rows(),
                [](double x) noexcept { return !std::isfinite(x); });
        });
    if (has_invalid)
        throw std::invalid_argument("Input matrix contains NaN or Inf");
}

void SVD::normalize_orientation_() {
    if (input_.rows() >= input_.cols()) {
        work_ = input_;
        transposed_ = false;
    } else {
        work_ = input_.transpose();
        transposed_ = true;
    }
}

void SVD::preprocess_qrcp_stage1_() {
    W1_ = work_;
    taus1_.assign(W1_.cols(), 0.0);
    perm1_.resize(W1_.cols());
    std::iota(perm1_.begin(), perm1_.end(), std::size_t{ 0 });
    businger_golub_inplace_(W1_, taus1_, perm1_);
}

void SVD::preprocess_qrcp_stage2_() {
    Matrix R1 = extract_R_(W1_);
    W2_ = R1.transpose();
    taus2_.assign(W2_.cols(), 0.0);
    qr_no_pivot_inplace_(W2_, taus2_);
}

void SVD::jacobi_on_R2_() {
    Matrix R2 = extract_R_(W2_);
    SVDResult svd_R2 = jacobi_engine_(R2);
    U_R2_ = std::move(svd_R2.U);
    V_R2_ = std::move(svd_R2.V);
    Sigma_ = std::move(svd_R2.Sigma);
}

SVDResult SVD::compose_thin_() {
    // R1 = (R1ᵀ)ᵀ = (Q2·R2)ᵀ = R2ᵀ·Q2ᵀ
    //    SVD of R2:  R2 = U_R2·Σ·V_R2ᵀ
    //    R2ᵀ = V_R2·Σ·U_R2ᵀ
    //    R1 = V_R2·Σ·U_R2ᵀ·Q2ᵀ  →  SVD of R1: U_R1 = V_R2,  V_R1 = Q2·U_R2
    // X = Q1·R1·P1ᵀ = (Q1·V_R2)·Σ·(P1·Q2·U_R2)ᵀ

    Matrix U_X = apply_Q_to_thin_(W1_, taus1_, V_R2_);

    Matrix Q2_U_R2 = U_R2_;
    apply_Q_left_inplace_(W2_, taus2_, Q2_U_R2);
    Matrix V_X = permute_rows_square_(perm1_, Q2_U_R2);

    return SVDResult(std::move(U_X), std::move(Sigma_), std::move(V_X));
}

SVDResult SVD::promote_to_full_(SVDResult thin) {
    const std::size_t M = input_.rows(), N = input_.cols();
    if (M > N) {
        Matrix U_full = extend_to_orthogonal_(thin.U);
        Matrix Sigma_full = make_padded_sigma_(M, N, thin.Sigma);
        return SVDResult(std::move(U_full), std::move(Sigma_full), std::move(thin.V));
    } else if (M < N) {
        Matrix V_full = extend_to_orthogonal_(thin.V);
        Matrix Sigma_full = make_padded_sigma_(M, N, thin.Sigma);
        return SVDResult(std::move(thin.U), std::move(Sigma_full), std::move(V_full));
    }
    return thin;
}

// =======================================================================
// [5] Static building blocks — Householder / QR primitives
// =======================================================================

SVD::Householder SVD::make_householder_(const double* x, std::size_t n) {
    if (n == 0) return { {}, 0.0, 0.0 };
    if (n == 1) return { { 1.0 }, 0.0, x[0] };

    const double xnorm_sq_rest = std::transform_reduce(std::execution::unseq,
        x + 1, x + n, 0.0, std::plus<>{},
        [](double v) noexcept { return v * v; });

    if (xnorm_sq_rest == 0.0) {
        return { std::vector<double>(n, 0.0), 0.0, x[0] };
    }

    const double xnorm = std::sqrt(x[0] * x[0] + xnorm_sq_rest);
    const double alpha = -std::copysign(xnorm, x[0]);
    const double v0 = x[0] - alpha;
    const double inv_v0 = 1.0 / v0;

    std::vector<double> v(n);
    v[0] = 1.0;
    for (std::size_t i = 1; i < n; ++i) v[i] = x[i] * inv_v0;

    double vtv = 1.0;
    for (std::size_t i = 1; i < n; ++i) vtv += v[i] * v[i];
    const double tau = 2.0 / vtv;

    return { std::move(v), tau, alpha };
}

void SVD::apply_householder_(const Householder& h, double* y) noexcept {
    if (h.tau == 0.0) return;
    const std::size_t n = h.v.size();
    double dot = 0.0;
    for (std::size_t i = 0; i < n; ++i) dot += h.v[i] * y[i];
    const double scale = h.tau * dot;
    for (std::size_t i = 0; i < n; ++i) y[i] -= scale * h.v[i];
}

void SVD::businger_golub_inplace_(Matrix& W, std::vector<double>& taus,
                                    std::vector<std::size_t>& perm) {
    const std::size_t M = W.rows(), N = W.cols();
    const std::size_t W_stride = W.stride_;
    double* W_data = W.data_.get();

    // 초기 열 norm²
    std::vector<double> col_norms_sq(N), original_norms_sq(N);
    {
        auto cols_iota = std::views::iota(std::size_t{ 0 }, N);
        std::for_each(std::execution::par_unseq, cols_iota.begin(), cols_iota.end(),
            [&](std::size_t j) noexcept {
                const double* col = W_data + j * W_stride;
                const double s = std::transform_reduce(std::execution::unseq,
                    col, col + M, 0.0, std::plus<>{},
                    [](double x) noexcept { return x * x; });
                col_norms_sq[j] = s;
                original_norms_sq[j] = s;
            });
    }

    for (std::size_t j = 0; j < N; ++j) {
        // 1. Pivot 선택
        std::size_t pivot = j;
        double max_norm_sq = col_norms_sq[j];
        for (std::size_t k = j + 1; k < N; ++k) {
            if (col_norms_sq[k] > max_norm_sq) {
                max_norm_sq = col_norms_sq[k];
                pivot = k;
            }
        }
        if (pivot != j) {
            double* col_j = W_data + j * W_stride;
            double* col_p = W_data + pivot * W_stride;
            std::swap_ranges(col_j, col_j + M, col_p);
            std::swap(col_norms_sq[j], col_norms_sq[pivot]);
            std::swap(original_norms_sq[j], original_norms_sq[pivot]);
            std::swap(perm[j], perm[pivot]);
        }

        // 2. Householder + W에 압축 저장
        double* col_j = W_data + j * W_stride;
        Householder h = make_householder_(col_j + j, M - j);
        taus[j] = h.tau;
        col_j[j] = h.alpha;
        for (std::size_t i = 1; i < M - j; ++i) col_j[j + i] = h.v[i];

        // 3. 남은 열에 H_j 적용 + norm downdate
        if (j + 1 < N) {
            auto rest = std::views::iota(j + 1, N);
            std::for_each(std::execution::par_unseq, rest.begin(), rest.end(),
                [&, j_local = j](std::size_t k) noexcept {
                    double* col_k = W_data + k * W_stride;
                    apply_householder_(h, col_k + j_local);

                    const double sub = col_k[j_local] * col_k[j_local];
                    double new_norm_sq = col_norms_sq[k] - sub;
                    if (new_norm_sq < 0.05 * original_norms_sq[k]) {
                        new_norm_sq = std::transform_reduce(std::execution::unseq,
                            col_k + j_local + 1, col_k + M, 0.0, std::plus<>{},
                            [](double x) noexcept { return x * x; });
                        original_norms_sq[k] = new_norm_sq;
                    }
                    col_norms_sq[k] = new_norm_sq;
                });
        }
    }
}

void SVD::qr_no_pivot_inplace_(Matrix& W, std::vector<double>& taus) {
    const std::size_t M = W.rows(), N = W.cols();
    const std::size_t W_stride = W.stride_;
    double* W_data = W.data_.get();

    for (std::size_t j = 0; j < N; ++j) {
        double* col_j = W_data + j * W_stride;
        Householder h = make_householder_(col_j + j, M - j);
        taus[j] = h.tau;
        col_j[j] = h.alpha;
        for (std::size_t i = 1; i < M - j; ++i) col_j[j + i] = h.v[i];

        if (j + 1 < N) {
            auto rest = std::views::iota(j + 1, N);
            std::for_each(std::execution::par_unseq, rest.begin(), rest.end(),
                [&, j_local = j](std::size_t k) noexcept {
                    double* col_k = W_data + k * W_stride;
                    apply_householder_(h, col_k + j_local);
                });
        }
    }
}

Matrix SVD::extract_R_(const Matrix& W) {
    const std::size_t N = W.cols();
    Matrix R(N, N, 0.0);
    for (std::size_t j = 0; j < N; ++j) {
        const double* col = W.data_.get() + j * W.stride_;
        for (std::size_t i = 0; i <= j; ++i) R(i, j) = col[i];
    }
    return R;
}

void SVD::apply_Q_left_inplace_(const Matrix& W, const std::vector<double>& taus,
                                  Matrix& target) {
    const std::size_t M = W.rows(), N_ref = W.cols(), K = target.cols();
    if (target.rows() != M)
        throw std::invalid_argument("apply_Q_left_inplace: target rows != W.rows");

    const std::size_t W_stride = W.stride_;
    const std::size_t T_stride = target.stride_;
    const double* W_data = W.data_.get();
    double* T_data = target.data_.get();

    for (std::size_t step = N_ref; step > 0; --step) {
        const std::size_t j = step - 1;
        const double tau_j = taus[j];
        if (tau_j == 0.0) continue;
        const double* w_col_j = W_data + j * W_stride;
        const std::size_t v_len = M - j;

        auto cols_iota = std::views::iota(std::size_t{ 0 }, K);
        std::for_each(std::execution::par_unseq, cols_iota.begin(), cols_iota.end(),
            [=](std::size_t k) noexcept {
                double* y = T_data + k * T_stride + j;
                double dot = y[0]; // v[0] = 1 implicit
                for (std::size_t i = 1; i < v_len; ++i) dot += w_col_j[j + i] * y[i];
                const double scale = tau_j * dot;
                y[0] -= scale;
                for (std::size_t i = 1; i < v_len; ++i) y[i] -= scale * w_col_j[j + i];
            });
    }
}

Matrix SVD::apply_Q_to_thin_(const Matrix& W, const std::vector<double>& taus,
                                const Matrix& U_R) {
    const std::size_t M = W.rows(), N = W.cols();
    if (U_R.rows() != N || U_R.cols() != N)
        throw std::invalid_argument("apply_Q_to_thin_: U_R must be N × N");

    Matrix result(M, N, 0.0);
    for (std::size_t j = 0; j < N; ++j) {
        const double* src = U_R.data_.get() + j * U_R.stride_;
        double* dst = result.data_.get() + j * result.stride_;
        std::copy_n(src, N, dst);
    }
    apply_Q_left_inplace_(W, taus, result);
    return result;
}

Matrix SVD::permute_rows_square_(const std::vector<std::size_t>& perm, const Matrix& V_in) {
    const std::size_t N = V_in.rows();
    if (V_in.cols() != N || perm.size() != N)
        throw std::invalid_argument("permute_rows_square_: dimension mismatch");

    Matrix V_out(N, N, Matrix::UninitializedTag{});
    const std::size_t in_stride = V_in.stride_;
    const std::size_t out_stride = V_out.stride_;
    const double* in_data = V_in.data_.get();
    double* out_data = V_out.data_.get();
    const std::size_t* perm_data = perm.data();

    auto cols_iota = std::views::iota(std::size_t{ 0 }, N);
    std::for_each(std::execution::par_unseq, cols_iota.begin(), cols_iota.end(),
        [=](std::size_t k) noexcept {
            const double* in_col = in_data + k * in_stride;
            double* out_col = out_data + k * out_stride;
            for (std::size_t l = 0; l < N; ++l) out_col[perm_data[l]] = in_col[l];
        });
    return V_out;
}

Matrix SVD::extend_to_orthogonal_(const Matrix& Q) {
    const std::size_t rows = Q.rows(), k = Q.cols();
    if (k > rows)
        throw std::invalid_argument("extend_to_orthogonal_: cols > rows");
    if (k == rows) return Q;

    Matrix Q_full(rows, rows, 0.0);
    const std::size_t Q_stride = Q.stride_;
    const std::size_t Qf_stride = Q_full.stride_;
    const double* Q_data = Q.data_.get();
    double* Qf_data = Q_full.data_.get();

    for (std::size_t j = 0; j < k; ++j) {
        const double* src = Q_data + j * Q_stride;
        double* dst = Qf_data + j * Qf_stride;
        std::copy_n(src, rows, dst);
    }

    const double accept_tol = std::sqrt(std::numeric_limits<double>::epsilon());
    std::vector<double> cand(rows);

    std::size_t added = k;
    for (std::size_t e = 0; e < rows && added < rows; ++e) {
        std::fill(cand.begin(), cand.end(), 0.0);
        cand[e] = 1.0;

        // MGS twice — Giraud, Langou, Rozložník 2005
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t j = 0; j < added; ++j) {
                const double* qj = Qf_data + j * Qf_stride;
                const double dot = std::transform_reduce(std::execution::unseq,
                    cand.begin(), cand.end(), qj, 0.0);
                for (std::size_t i = 0; i < rows; ++i) cand[i] -= dot * qj[i];
            }
        }

        const double norm_sq = std::transform_reduce(std::execution::unseq,
            cand.begin(), cand.end(), 0.0, std::plus<>{},
            [](double x) noexcept { return x * x; });
        const double norm = std::sqrt(norm_sq);

        if (norm > accept_tol) {
            const double inv = 1.0 / norm;
            double* dst = Qf_data + added * Qf_stride;
            for (std::size_t i = 0; i < rows; ++i) dst[i] = cand[i] * inv;
            ++added;
        }
    }

    if (added != rows)
        throw std::runtime_error("orthogonal extension failed: insufficient basis vectors");
    return Q_full;
}

Matrix SVD::make_padded_sigma_(std::size_t target_M, std::size_t target_N,
                                  const Matrix& Sigma_diag) {
    Matrix S(target_M, target_N, 0.0);
    const std::size_t k = std::min({ target_M, target_N, Sigma_diag.rows(), Sigma_diag.cols() });
    for (std::size_t i = 0; i < k; ++i) S(i, i) = Sigma_diag(i, i);
    return S;
}

// =======================================================================
// [6] Hestenes one-sided Jacobi engine
//     R2 (작고 잘 조건화된 N×N) 위에서 직접 SVD 계산.
// =======================================================================

SVDResult SVD::jacobi_engine_(const Matrix& A) {
    const std::size_t M = A.rows(), N = A.cols();
    Matrix U = A;
    Matrix V(N, N, 0.0);

    const std::size_t U_stride = U.stride_;
    const std::size_t V_stride = V.stride_;
    const std::size_t A_stride = A.stride_;
    double* U_data = U.data_.get();
    double* V_data = V.data_.get();
    const double* A_data = A.data_.get();

    auto N_indices = std::views::iota(std::size_t{ 0 }, N);
    auto M_indices = std::views::iota(std::size_t{ 0 }, M);

    std::for_each(std::execution::unseq, N_indices.begin(), N_indices.end(),
        [&](std::size_t i) noexcept { V(i, i) = 1.0; });

    if (N <= 1) return SVDResult(std::move(U), Matrix(N, N, 0.0), std::move(V));

    // ── Column pivoting (norm-based, descending) ──
    {
        std::vector<double> col_norms_sq(N);
        std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(),
            [&](std::size_t j) noexcept {
                const double* u_ptr = std::assume_aligned<64>(U_data + j * U_stride);
                col_norms_sq[j] = std::transform_reduce(std::execution::unseq,
                    u_ptr, u_ptr + M, 0.0, std::plus<>{},
                    [](double x) noexcept { return x * x; });
            });

        std::vector<std::size_t> piv(N);
        std::iota(piv.begin(), piv.end(), 0);
        std::sort(piv.begin(), piv.end(), [&](std::size_t a, std::size_t b) noexcept {
            return col_norms_sq[a] > col_norms_sq[b];
        });

        bool needs_pivot = false;
        for (std::size_t j = 0; j < N; ++j) { if (piv[j] != j) { needs_pivot = true; break; } }

        if (needs_pivot) {
            Matrix U_piv(M, N, Matrix::UninitializedTag{});
            const std::size_t Up_stride = U_piv.stride_;
            double* Up_data = U_piv.data_.get();
            for (std::size_t j = 0; j < N; ++j) {
                const double* src = std::assume_aligned<64>(U_data + piv[j] * U_stride);
                double* dst = std::assume_aligned<64>(Up_data + j * Up_stride);
                std::copy_n(src, M, dst);
            }
            U = std::move(U_piv);
            U_data = U.data_.get();
            std::memset(V_data, 0, V_stride * V.cols() * sizeof(double));
            for (std::size_t j = 0; j < N; ++j) V(piv[j], j) = 1.0;
        }
    }

    constexpr int MAX_SWEEPS = 30;
    const double tol = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(M, N));

    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4;
    const bool par_outer = (N >= std::max(64u, num_cores * 8));
    const bool par_inner = !par_outer && (M >= 1024);

    const std::size_t n_even = N + (N % 2);
    auto ptr_machines = std::make_unique<std::size_t[]>(n_even);
    std::span<std::size_t> machines(ptr_machines.get(), n_even);
    std::iota(machines.begin(), machines.end(), 0);

    auto ptr_pairs = std::make_unique<std::pair<std::size_t, std::size_t>[]>(n_even / 2);
    std::span<std::pair<std::size_t, std::size_t>> pairs_buffer(ptr_pairs.get(), n_even / 2);

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double sweep_max_off = 0.0;
        // Threshold Jacobi
        const double sweep_threshold = std::max(tol, std::pow(0.1, sweep + 1));

        for (std::size_t step = 0; step < n_even - 1; ++step) {
            std::size_t valid_pairs = 0;
            for (std::size_t k = 0; k < n_even / 2; ++k) {
                std::size_t c1 = machines[k], c2 = machines[n_even - 1 - k];
                if (c1 < N && c2 < N)
                    pairs_buffer[valid_pairs++] = (c1 < c2) ? std::make_pair(c1, c2) : std::make_pair(c2, c1);
            }
            auto active_pairs = pairs_buffer.subspan(0, valid_pairs);

            auto process_pair = [&](const std::pair<std::size_t, std::size_t>& p_idx) noexcept -> double {
                const std::size_t i = p_idx.first, j = p_idx.second;

                double* ui_ptr = std::assume_aligned<64>(U_data + i * U_stride);
                double* uj_ptr = std::assume_aligned<64>(U_data + j * U_stride);

                PQR pqr = (par_inner)
                    ? std::transform_reduce(std::execution::par_unseq, ui_ptr, ui_ptr + M, uj_ptr,
                        PQR{ 0.0, 0.0, 0.0 }, pqr_combine, pqr_map)
                    : std::transform_reduce(std::execution::unseq, ui_ptr, ui_ptr + M, uj_ptr,
                        PQR{ 0.0, 0.0, 0.0 }, pqr_combine, pqr_map);

                const double p = pqr.p, q = pqr.q, r = pqr.r;
                const double sq = std::sqrt(q), sr = std::sqrt(r);
                const double off_diag = (sq == 0.0 || sr == 0.0) ? 0.0 : std::abs(p) / (sq * sr);
                if (off_diag <= sweep_threshold) return off_diag;

                const double theta = (r - q) / (2.0 * p);
                double t;
                if (std::isinf(theta)) {
                    t = 0.0;
                } else {
                    t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(1.0, theta));
                }
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = c * t;

                double* vi_ptr = std::assume_aligned<64>(V_data + i * V_stride);
                double* vj_ptr = std::assume_aligned<64>(V_data + j * V_stride);

                auto apply_rot = [c, s](double& xi, double& xj) noexcept {
                    const double vi = xi, vj = xj;
                    xi = c * vi - s * vj;
                    xj = s * vi + c * vj;
                };

                if (par_inner) {
                    std::for_each(std::execution::par_unseq, M_indices.begin(), M_indices.end(),
                        [&](std::size_t idx) noexcept { apply_rot(ui_ptr[idx], uj_ptr[idx]); });
                    std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(),
                        [&](std::size_t idx) noexcept { apply_rot(vi_ptr[idx], vj_ptr[idx]); });
                } else {
                    std::for_each(std::execution::unseq, M_indices.begin(), M_indices.end(),
                        [&](std::size_t idx) noexcept { apply_rot(ui_ptr[idx], uj_ptr[idx]); });
                    std::for_each(std::execution::unseq, N_indices.begin(), N_indices.end(),
                        [&](std::size_t idx) noexcept { apply_rot(vi_ptr[idx], vj_ptr[idx]); });
                }
                return off_diag;
            };

            auto max_op = [](double a, double b) constexpr noexcept { return a > b ? a : b; };
            const double step_max_off = (par_outer)
                ? std::transform_reduce(std::execution::par_unseq, active_pairs.begin(), active_pairs.end(),
                    0.0, max_op, process_pair)
                : std::transform_reduce(std::execution::seq, active_pairs.begin(), active_pairs.end(),
                    0.0, max_op, process_pair);

            if (step_max_off > sweep_max_off) sweep_max_off = step_max_off;
            std::rotate(machines.begin() + 1, machines.end() - 1, machines.end());
        }
        if (sweep_max_off <= tol) break;
    }

    // ── 수렴 후 V 직교성 검증 + 필요 시 MGS twice 재직교화 ──
    {
        const double ortho_tol = std::sqrt(std::numeric_limits<double>::epsilon());
        double max_ortho_err = 0.0;
        bool needs_reorth = false;

        for (std::size_t i = 0; i < N && !needs_reorth; ++i) {
            const double* vi_ptr = std::assume_aligned<64>(V_data + i * V_stride);
            for (std::size_t j = i + 1; j < N; ++j) {
                const double* vj_ptr = std::assume_aligned<64>(V_data + j * V_stride);
                const double dot = std::transform_reduce(std::execution::unseq,
                    vi_ptr, vi_ptr + N, vj_ptr, 0.0);
                const double abs_dot = std::abs(dot);
                if (abs_dot > max_ortho_err) max_ortho_err = abs_dot;
                if (max_ortho_err > ortho_tol) { needs_reorth = true; break; }
            }
        }

        if (needs_reorth) {
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t j = 0; j < N; ++j) {
                    double* vj_ptr = V_data + j * V_stride;
                    for (std::size_t k = 0; k < j; ++k) {
                        const double* vk_ptr = std::assume_aligned<64>(V_data + k * V_stride);
                        const double proj = std::transform_reduce(std::execution::unseq,
                            vj_ptr, vj_ptr + N, vk_ptr, 0.0);
                        for (std::size_t idx = 0; idx < N; ++idx) vj_ptr[idx] -= proj * vk_ptr[idx];
                    }
                    const double norm_sq = std::transform_reduce(std::execution::unseq,
                        vj_ptr, vj_ptr + N, 0.0, std::plus<>{},
                        [](double x) noexcept { return x * x; });
                    const double norm = std::sqrt(norm_sq);
                    if (norm > std::numeric_limits<double>::min()) {
                        const double inv = 1.0 / norm;
                        std::for_each(std::execution::unseq, vj_ptr, vj_ptr + N,
                            [inv](double& x) noexcept { x *= inv; });
                    }
                }
            }

            // U = A · V 재계산
            std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(),
                [&](std::size_t j) noexcept {
                    double* uj_ptr = std::assume_aligned<64>(U_data + j * U_stride);
                    std::fill_n(uj_ptr, M, 0.0);
                    const double* vj_ptr = std::assume_aligned<64>(V_data + j * V_stride);
                    for (std::size_t k = 0; k < N; ++k) {
                        const double v_kj = vj_ptr[k];
                        if (v_kj == 0.0) continue;
                        const double* ak_ptr = std::assume_aligned<64>(A_data + k * A_stride);
                        for (std::size_t i = 0; i < M; ++i) uj_ptr[i] += v_kj * ak_ptr[i];
                    }
                });
        }
    }

    // ── 특이값 추출 + 정렬 ──
    std::vector<double> S_vec(N);
    std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(),
        [&](std::size_t j) noexcept {
            double* u_ptr = std::assume_aligned<64>(U_data + j * U_stride);
            const double norm_sq = std::transform_reduce(std::execution::unseq,
                u_ptr, u_ptr + M, 0.0, std::plus<>{},
                [](double x) noexcept { return x * x; });
            S_vec[j] = std::sqrt(norm_sq);
        });

    const double max_S = *std::max_element(std::execution::unseq, S_vec.begin(), S_vec.end());
    const double rank_tol = std::max(
        std::numeric_limits<double>::epsilon() * static_cast<double>(M) * max_S,
        std::numeric_limits<double>::min());

    for (std::size_t j = 0; j < N; ++j) {
        double* u_ptr = std::assume_aligned<64>(U_data + j * U_stride);
        if (S_vec[j] > rank_tol) {
            const double inv_sig = 1.0 / S_vec[j];
            std::for_each(std::execution::unseq, u_ptr, u_ptr + M,
                [inv_sig](double& x) noexcept { x *= inv_sig; });
        } else {
            S_vec[j] = 0.0;
            std::for_each(std::execution::unseq, u_ptr, u_ptr + M,
                [](double& x) noexcept { x = 0.0; });
        }
    }

    std::vector<std::size_t> p(N);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t a, std::size_t b) noexcept { return S_vec[a] > S_vec[b]; });

    Matrix U_final(M, N, Matrix::UninitializedTag{});
    Matrix V_final(N, N, Matrix::UninitializedTag{});
    Matrix Sigma_final(N, N, 0.0);

    const std::size_t Uf_stride = U_final.stride_;
    const std::size_t Vf_stride = V_final.stride_;
    double* Uf_data = U_final.data_.get();
    double* Vf_data = V_final.data_.get();

    std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(),
        [&](std::size_t new_j) noexcept {
            const std::size_t old_j = p[new_j];
            Sigma_final(new_j, new_j) = S_vec[old_j];
            const double* u_src = std::assume_aligned<64>(U_data + old_j * U_stride);
            double* u_dst = std::assume_aligned<64>(Uf_data + new_j * Uf_stride);
            std::copy(std::execution::unseq, u_src, u_src + M, u_dst);

            const double* v_src = std::assume_aligned<64>(V_data + old_j * V_stride);
            double* v_dst = std::assume_aligned<64>(Vf_data + new_j * Vf_stride);
            std::copy(std::execution::unseq, v_src, v_src + N, v_dst);
        });

    return SVDResult(std::move(U_final), std::move(Sigma_final), std::move(V_final));
}

// =======================================================================
// [7] matmul — 검증·시연 유틸리티
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
