#pragma once
//
// 내부: compute_svd가 사용하는 보조 함수들.
//   - extend_to_orthogonal: thin Q (rows × k)을 정사각 직교 행렬로 확장 (Full SVD용)
//   - make_padded_sigma: 특이값을 target M×N Σ로 패딩
//   - thin_svd_with_qrcp: 2단계 QRCP 전처리 + Jacobi + 결과 합성
//

#include "../svd.hpp"
#include "allocator.hpp"
#include "qrcp.hpp"
#include "jacobi.hpp"

#include <cstddef>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <execution>
#include <ranges>
#include <stdexcept>

namespace svd::detail {

// 정규직교 열 k개를 가진 Q (rows×k)을 rows×rows 정사각 직교 행렬로 확장.
// 표준 기저 e_0..e_{rows-1}을 차례로 시도하고 MGS twice로 직교화 후 채택.
inline Matrix extend_to_orthogonal(const Matrix& Q) {
    const std::size_t rows = Q.rows(), k = Q.cols();
    if (k > rows)
        throw std::invalid_argument("extend_to_orthogonal: cols > rows");
    if (k == rows) return Q;

    Matrix Q_full(rows, rows, 0.0);
    const std::size_t Q_stride = MatrixAccess::stride(Q);
    const std::size_t Qf_stride = MatrixAccess::stride(Q_full);
    const double* Q_data = MatrixAccess::data(Q);
    double* Qf_data = MatrixAccess::data(Q_full);

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

// k = min(target_M, target_N) 개의 특이값을 target_M × target_N Σ로 패딩.
inline Matrix make_padded_sigma(std::size_t target_M, std::size_t target_N, const Matrix& Sigma_diag) {
    Matrix S(target_M, target_N, 0.0);
    const std::size_t k = std::min({ target_M, target_N, Sigma_diag.rows(), Sigma_diag.cols() });
    for (std::size_t i = 0; i < k; ++i) S(i, i) = Sigma_diag(i, i);
    return S;
}

// 2단계 QRCP 전처리 + Jacobi (Drmač-Veselić §3.5).
//   Stage 1: X·P1 = Q1·R1                (column-pivoting QRCP)
//   Stage 2: R1ᵀ = Q2·R2                 (plain QR; R1ᵀ는 이미 잘 sorted)
//   Jacobi : R2 = U_R2·Σ·V_R2ᵀ
//   합성   : X = (Q1·V_R2)·Σ·(P1·Q2·U_R2)ᵀ
inline SVDResult thin_svd_with_qrcp(const Matrix& X) {
    const std::size_t N = X.cols();
    if (N <= 1) return svd_tall(X);

    QRCPCompact qr1 = businger_golub_qrcp(X);
    Matrix R1 = extract_R_from_compact(qr1.W);

    Matrix R1_T = R1.transpose();
    QRCompact qr2 = qr_no_pivot(R1_T);
    Matrix R2 = extract_R_from_compact(qr2.W);

    SVDResult svd_R2 = svd_tall(R2);

    // R1 = (Q2·R2)ᵀ = R2ᵀ·Q2ᵀ = (V_R2·Σ·U_R2ᵀ)·Q2ᵀ
    //  → SVD of R1: U_R1 = V_R2,  V_R1 = Q2·U_R2
    // X = Q1·R1·P1ᵀ = (Q1·V_R2)·Σ·(P1·Q2·U_R2)ᵀ
    Matrix U_X = apply_Q_to_thin_target(qr1.W, qr1.taus, svd_R2.V);

    Matrix Q2_U_R2 = svd_R2.U;
    apply_Q_left_inplace(qr2.W, qr2.taus, Q2_U_R2);
    Matrix V_X = permute_rows_square(qr1.perm, Q2_U_R2);

    return SVDResult(std::move(U_X), std::move(svd_R2.Sigma), std::move(V_X));
}

} // namespace svd::detail
