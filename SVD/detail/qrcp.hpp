#pragma once
//
// 내부: Businger-Golub QRCP (column pivoting) + plain QR.
// Implicit Q storage — Householder essential v[1..]는 W의 sub-diagonal에 압축.
// 메모리: 기존 명시적 Q 대비 ~3배 절감.
//

#include "../svd.hpp"
#include "allocator.hpp"      // MatrixAccess
#include "householder.hpp"

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <execution>
#include <ranges>

namespace svd::detail {

// QRCP 결과 — implicit Q.
struct QRCPCompact {
    Matrix W;                      // M × N: upper triangle = R, sub-diagonal = essential v[1..]
    std::vector<double> taus;      // length N
    std::vector<std::size_t> perm; // perm[j] = R의 j번째 열에 들어간 원본 A 열 index
};

// Plain QR (no pivoting) 결과 — implicit Q. 2단계 QRCP의 stage 2에서 사용.
struct QRCompact {
    Matrix W;
    std::vector<double> taus;
};

// W의 upper triangle에서 N×N 상삼각 R을 복사 추출.
inline Matrix extract_R_from_compact(const Matrix& W) {
    const std::size_t N = W.cols();
    Matrix R(N, N, 0.0);
    for (std::size_t j = 0; j < N; ++j) {
        const double* col = MatrixAccess::data(W) + j * MatrixAccess::stride(W);
        for (std::size_t i = 0; i <= j; ++i) R(i, j) = col[i];
    }
    return R;
}

// target ← Q · target,  Q = H_0·H_1·…·H_{N_ref-1} (W에 implicit 저장).
// target.rows == W.rows 가정. 작용 순서는 H_{N_ref-1}, …, H_0 (역순).
inline void apply_Q_left_inplace(const Matrix& W, const std::vector<double>& taus,
    Matrix& target) {
    const std::size_t M = W.rows(), N_ref = W.cols(), K = target.cols();
    if (target.rows() != M)
        throw std::invalid_argument("apply_Q_left_inplace: target rows != W.rows");

    const std::size_t W_stride = MatrixAccess::stride(W);
    const std::size_t T_stride = MatrixAccess::stride(target);
    const double* W_data = MatrixAccess::data(W);
    double* T_data = MatrixAccess::data(target);

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
                // dot = v[0]·y[0] + Σ v[i]·y[i],  v[0] = 1 (implicit)
                double dot = y[0];
                for (std::size_t i = 1; i < v_len; ++i) dot += w_col_j[j + i] * y[i];
                const double scale = tau_j * dot;
                y[0] -= scale;
                for (std::size_t i = 1; i < v_len; ++i) y[i] -= scale * w_col_j[j + i];
            });
    }
}

// implicit Q (M × N_ref)를 N_ref × N_ref U_R에 적용 → M × N_ref 결과.
inline Matrix apply_Q_to_thin_target(const Matrix& W, const std::vector<double>& taus,
    const Matrix& U_R) {
    const std::size_t M = W.rows(), N = W.cols();
    if (U_R.rows() != N || U_R.cols() != N)
        throw std::invalid_argument("apply_Q_to_thin_target: U_R must be N × N");

    Matrix result(M, N, 0.0);
    for (std::size_t j = 0; j < N; ++j) {
        const double* src = MatrixAccess::data(U_R) + j * MatrixAccess::stride(U_R);
        double* dst = MatrixAccess::data(result) + j * MatrixAccess::stride(result);
        std::copy_n(src, N, dst);
    }
    apply_Q_left_inplace(W, taus, result);
    return result;
}

// V_out[perm[l], k] = V_in[l, k]  (V_out = P · V_in)
inline Matrix permute_rows_square(const std::vector<std::size_t>& perm, const Matrix& V_in) {
    const std::size_t N = V_in.rows();
    if (V_in.cols() != N || perm.size() != N)
        throw std::invalid_argument("permute_rows_square: dimension mismatch");

    Matrix V_out(N, N, Matrix::UninitializedTag{});
    const std::size_t in_stride = MatrixAccess::stride(V_in);
    const std::size_t out_stride = MatrixAccess::stride(V_out);
    const double* in_data = MatrixAccess::data(V_in);
    double* out_data = MatrixAccess::data(V_out);
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

// Businger-Golub QR with column pivoting.  A·P = Q·R,  |R(0,0)| ≥ |R(1,1)| ≥ …
inline QRCPCompact businger_golub_qrcp(const Matrix& A) {
    if (A.rows() < A.cols())
        throw std::invalid_argument("businger_golub_qrcp requires rows >= cols");

    const std::size_t M = A.rows(), N = A.cols();
    Matrix W = A;
    const std::size_t W_stride = MatrixAccess::stride(W);
    double* W_data = MatrixAccess::data(W);

    std::vector<std::size_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<double> taus(N, 0.0);

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
        HouseholderResult h = make_householder(col_j + j, M - j);
        taus[j] = h.tau;
        col_j[j] = h.alpha;
        for (std::size_t i = 1; i < M - j; ++i)
            col_j[j + i] = h.v[i]; // v[1..] sub-diagonal에 저장 (v[0]=1 implicit)

        // 3. 남은 열들에 H_j 적용 + norm downdate
        if (j + 1 < N) {
            auto rest = std::views::iota(j + 1, N);
            std::for_each(std::execution::par_unseq, rest.begin(), rest.end(),
                [&, j_local = j](std::size_t k) noexcept {
                    double* col_k = W_data + k * W_stride;
                    apply_householder(h, col_k + j_local);

                    const double sub = col_k[j_local] * col_k[j_local];
                    double new_norm_sq = col_norms_sq[k] - sub;
                    // LAPACK 휴리스틱: cancellation 심하면 재계산
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

    return { std::move(W), std::move(taus), std::move(perm) };
}

// Plain Householder QR (no pivoting).  2단계 QRCP의 stage 2에 사용.
inline QRCompact qr_no_pivot(const Matrix& A) {
    if (A.rows() < A.cols())
        throw std::invalid_argument("qr_no_pivot requires rows >= cols");

    const std::size_t M = A.rows(), N = A.cols();
    Matrix W = A;
    const std::size_t W_stride = MatrixAccess::stride(W);
    double* W_data = MatrixAccess::data(W);
    std::vector<double> taus(N, 0.0);

    for (std::size_t j = 0; j < N; ++j) {
        double* col_j = W_data + j * W_stride;
        HouseholderResult h = make_householder(col_j + j, M - j);
        taus[j] = h.tau;
        col_j[j] = h.alpha;
        for (std::size_t i = 1; i < M - j; ++i) col_j[j + i] = h.v[i];

        if (j + 1 < N) {
            auto rest = std::views::iota(j + 1, N);
            std::for_each(std::execution::par_unseq, rest.begin(), rest.end(),
                [&, j_local = j](std::size_t k) noexcept {
                    double* col_k = W_data + k * W_stride;
                    apply_householder(h, col_k + j_local);
                });
        }
    }
    return { std::move(W), std::move(taus) };
}

} // namespace svd::detail
