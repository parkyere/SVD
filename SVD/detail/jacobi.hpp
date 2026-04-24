#pragma once
//
// 내부: Hestenes one-sided Jacobi SVD engine (svd_tall).
// Pre: A.rows >= A.cols.  Returns thin SVD: U (M×N), Σ (N×N), V (N×N).
//
// 특징:
//   - 초기 column pivoting (norm-based)
//   - Round-robin parallel pair scheduling (par_unseq 안전)
//   - Threshold Jacobi (sweep마다 회전 cutoff 동적 조정)
//   - q*r underflow 회피 (sqrt 분리)
//   - 수렴 후 V 직교성 검증 + MGS twice 재직교화
//   - 절대 하한 포함 rank threshold
//

#include "../svd.hpp"
#include "allocator.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <memory>
#include <thread>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <execution>
#include <ranges>
#include <cstring>
#include <utility>

namespace svd::detail {

// PQR fused reduction — column pair (i, j)에 대해 한 pass로 (vᵢᵀvⱼ, ||vᵢ||², ||vⱼ||²) 누적.
struct PQR { double p, q, r; };
inline constexpr auto pqr_combine = [](const PQR& a, const PQR& b) noexcept -> PQR {
    return { a.p + b.p, a.q + b.q, a.r + b.r };
};
inline constexpr auto pqr_map = [](double vi, double vj) noexcept -> PQR {
    return { vi * vj, vi * vi, vj * vj };
};

inline SVDResult svd_tall(const Matrix& A) {
    const std::size_t M = A.rows(), N = A.cols();
    Matrix U = A;
    Matrix V(N, N, 0.0);

    const std::size_t U_stride = MatrixAccess::stride(U);
    const std::size_t V_stride = MatrixAccess::stride(V);
    const std::size_t A_stride = MatrixAccess::stride(A);
    double* U_data = MatrixAccess::data(U);
    double* V_data = MatrixAccess::data(V);
    const double* A_data = MatrixAccess::data(A);

    auto N_indices = std::views::iota(std::size_t{ 0 }, N);
    auto M_indices = std::views::iota(std::size_t{ 0 }, M);

    std::for_each(std::execution::unseq, N_indices.begin(), N_indices.end(),
        [&](std::size_t i) noexcept { V(i, i) = 1.0; });

    if (N <= 1) return SVDResult(std::move(U), Matrix(N, N, 0.0), std::move(V));

    // ── Column pivoting (norm-based, descending) ──────────────────────
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
            const std::size_t Up_stride = MatrixAccess::stride(U_piv);
            double* Up_data = MatrixAccess::data(U_piv);
            for (std::size_t j = 0; j < N; ++j) {
                const double* src = std::assume_aligned<64>(U_data + piv[j] * U_stride);
                double* dst = std::assume_aligned<64>(Up_data + j * Up_stride);
                std::copy_n(src, M, dst);
            }
            U = std::move(U_piv);
            U_data = MatrixAccess::data(U);
            // U_stride는 동일 dimensions이므로 변하지 않음
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

        // Threshold Jacobi: 초반 sweep에선 작은 회전 skip, 점진적으로 tol에 수렴
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
                // q*r 직접 곱 underflow 회피
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

    // ── 수렴 후 V 직교성 검증 + 필요 시 MGS twice 재직교화 ──────────
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
            // "Twice is enough" (Giraud, Langou, Rozložník 2005)
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

            // U = A · V 재계산 (열 기반 DAXPY 누적)
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

    // ── 특이값 추출 + 정렬 ───────────────────────────────────────────
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

    const std::size_t Uf_stride = MatrixAccess::stride(U_final);
    const std::size_t Vf_stride = MatrixAccess::stride(V_final);
    double* Uf_data = MatrixAccess::data(U_final);
    double* Vf_data = MatrixAccess::data(V_final);

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

} // namespace svd::detail
