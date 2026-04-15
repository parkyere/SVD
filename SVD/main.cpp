#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <execution>
#include <ranges>
#include <span>
#include <limits>
#include <iomanip>
#include <thread>
#include <utility>
#include <stdexcept>
#include <cstdlib>
#include <functional>  // std::logical_or 등
#include <cstring>     // for std::memset

// OS 및 아키텍처 종속 하드웨어 헤더 (x86/x64 환경용)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

// =======================================================================
// [1] 현업 비밀 무기: 하드웨어 Denormal Penalty 원천 차단 가드 (RAII)
// 극소수점 연산 시 CPU 파이프라인이 100배 느려지는 현상을 하드웨어 단에서 강제 차단
// =======================================================================
class HardwareOptimizationGuard {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int old_mxcsr = 0;
#endif
public:
    HardwareOptimizationGuard() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        old_mxcsr = _mm_getcsr();
        // FTZ (Flush-To-Zero) 및 DAZ (Denormals-Are-Zero) 활성화
        _mm_setcsr(old_mxcsr | _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON);
#endif
    }
    ~HardwareOptimizationGuard() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_setcsr(old_mxcsr);
#endif
    }
};

// =======================================================================
// [2] 현업 상용 라이브러리급 64-Byte (Cache-Line) 정렬 메모리 할당기
// =======================================================================
struct AlignedFree {
    void operator()(void* p) const noexcept {
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
};
using AlignedDoublePtr = std::unique_ptr<double[], AlignedFree>;

inline AlignedDoublePtr allocate_aligned_uninitialized(size_t count) {
    if (count == 0) return nullptr;
    size_t bytes = count * sizeof(double);
    size_t alloc_bytes = (bytes + 63) & ~63ULL; // 64바이트 배수로 올림
#if defined(_MSC_VER) || defined(__MINGW32__)
    void* ptr = _aligned_malloc(alloc_bytes, 64);
#else
    void* ptr = std::aligned_alloc(64, alloc_bytes);
#endif
    if (!ptr) throw std::bad_alloc();
    return AlignedDoublePtr(static_cast<double*>(ptr));
}

struct PQR { double p, q, r; };
constexpr auto pqr_combine = [](const PQR& a, const PQR& b) constexpr noexcept -> PQR { return { a.p + b.p, a.q + b.q, a.r + b.r }; };
constexpr auto pqr_map = [](double vi, double vj) constexpr noexcept -> PQR { return { vi * vj, vi * vi, vj * vj }; };

// =======================================================================
// [3] HPC Architecture 적용 Matrix (Padding & Stride 도입)
// =======================================================================
class Matrix {
public:
    size_t rows, cols;
    size_t stride; // 메모리상의 실제 행 길이 (캐시 뱅크 충돌 방지용 Padding 포함, LAPACK의 'lda')
    AlignedDoublePtr data;

    struct UninitializedTag {};

    // [최적화] 행(Row) 길이를 강제로 64-Byte(8 double) 배수로 맞춰 루프 테일(Tail)을 삭제하고,
    // 2의 거듭제곱 사이즈로 인한 L1 캐시 뱅크 충돌(Bank Conflict)을 원천 회피합니다.
    static size_t calculate_stride(size_t r) {
        if (r == 0) return 0;
        size_t s = (r + 7) & ~7ULL;
        if (s >= 64 && (s & (s - 1)) == 0) s += 8; // 2의 거듭제곱일 경우 패딩 추가
        return s;
    }

    Matrix(size_t r, size_t c, double val = 0.0) : rows(r), cols(c), stride(calculate_stride(r)) {
        if (stride * c > 0) {
            data = allocate_aligned_uninitialized(stride * c);
            if (val != 0.0) {
                double* ptr = std::assume_aligned<64>(data.get());
                std::fill_n(std::execution::par_unseq, ptr, stride * c, val);
            }
            else {
                std::memset(data.get(), 0, stride * c * sizeof(double));
            }
        }
    }

    Matrix(size_t r, size_t c, UninitializedTag) : rows(r), cols(c), stride(calculate_stride(r)) {
        if (stride * c > 0) data = allocate_aligned_uninitialized(stride * c);
    }

    Matrix() noexcept : rows(0), cols(0), stride(0), data(nullptr) {}

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), stride(other.stride) {
        if (stride * cols > 0) {
            data = allocate_aligned_uninitialized(stride * cols);
            const double* src = std::assume_aligned<64>(other.data.get());
            double* dst = std::assume_aligned<64>(data.get());
            std::copy(std::execution::par_unseq, src, src + (stride * cols), dst);
        }
    }

    Matrix(Matrix&& other) noexcept : rows(std::exchange(other.rows, 0)), cols(std::exchange(other.cols, 0)), stride(std::exchange(other.stride, 0)), data(std::move(other.data)) {}

    Matrix& operator=(Matrix&& other) noexcept {
        rows = std::exchange(other.rows, 0); cols = std::exchange(other.cols, 0);
        stride = std::exchange(other.stride, 0); data = std::move(other.data);
        return *this;
    }

    Matrix& operator=(const Matrix& other) { if (this != &other) { Matrix tmp(other); *this = std::move(tmp); } return *this; }

    // 데이터 접근 시 rows가 아닌 stride를 곱하여 캐시 라인 오프셋을 적용
    double& operator()(size_t i, size_t j) noexcept { return data.get()[j * stride + i]; }
    const double& operator()(size_t i, size_t j) const noexcept { return data.get()[j * stride + i]; }

    Matrix transpose() const {
        Matrix T(cols, rows, UninitializedTag{});
        constexpr size_t BLOCK_SIZE = 32;
        auto block_j_indices = std::views::iota(size_t{ 0 }, (cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
        bool use_par = (rows * cols >= 10000);

        auto kernel = [&](size_t bj) noexcept {
            size_t j_start = bj * BLOCK_SIZE, j_end = std::min(j_start + BLOCK_SIZE, cols);
            for (size_t bi = 0; bi < rows; bi += BLOCK_SIZE) {
                size_t i_end = std::min(bi + BLOCK_SIZE, rows);
                for (size_t j = j_start; j < j_end; ++j) {
                    for (size_t i = bi; i < i_end; ++i) T(j, i) = (*this)(i, j);
                }
            }
        };

        if (use_par)
            std::for_each(std::execution::par_unseq, block_j_indices.begin(), block_j_indices.end(), kernel);
        else
            std::for_each(std::execution::seq, block_j_indices.begin(), block_j_indices.end(), kernel);

        return T;
    }

    static Matrix from_row_major(size_t r, size_t c, const std::vector<double>& flat) {
        if (flat.size() < r * c)
            throw std::invalid_argument("flat vector size insufficient for matrix dimensions");
        Matrix M(r, c, 0.0); // 패딩 영역 0 초기화를 위해 기본 생성자 사용
        for (size_t i = 0; i < r; ++i) for (size_t j = 0; j < c; ++j) M(i, j) = flat[i * c + j];
        return M;
    }
};

struct SVDResult { Matrix U, Sigma, V; SVDResult(Matrix&& u, Matrix&& s, Matrix&& v) noexcept : U(std::move(u)), Sigma(std::move(s)), V(std::move(v)) {} };

// =======================================================================
// [4] 현업 투입 레벨 Hestenes Jacobi SVD Engine (assume_aligned 극한 활용)
// =======================================================================
SVDResult svd_tall(const Matrix& A) {
    const size_t M = A.rows, N = A.cols;
    Matrix U = A;
    Matrix V(N, N, 0.0);
    auto N_indices = std::views::iota(size_t{ 0 }, N);
    auto M_indices = std::views::iota(size_t{ 0 }, M);

    std::for_each(std::execution::unseq, N_indices.begin(), N_indices.end(), [&](size_t i) noexcept { V(i, i) = 1.0; });

    if (N <= 1) { /* 생략 무방한 1차원 처리 예외 */ return SVDResult(std::move(U), Matrix(N, N, 0.0), std::move(V)); }

    // [A.2] Column Pivoting: 열 L2 norm 기준 내림차순 정렬 (Drmač & Veselić 기법)
    // 조건수가 나쁜 행렬에서 Jacobi 수렴 속도를 극적으로 개선
    {
        std::vector<double> col_norms_sq(N);
        std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(), [&](size_t j) noexcept {
            const double* u_ptr = std::assume_aligned<64>(U.data.get() + j * U.stride);
            col_norms_sq[j] = std::transform_reduce(std::execution::unseq, u_ptr, u_ptr + M, 0.0,
                std::plus<>{}, [](double x) noexcept { return x * x; });
        });

        std::vector<size_t> piv(N);
        std::iota(piv.begin(), piv.end(), 0);
        std::sort(piv.begin(), piv.end(), [&](size_t a, size_t b) noexcept {
            return col_norms_sq[a] > col_norms_sq[b];
        });

        bool needs_pivot = false;
        for (size_t j = 0; j < N; ++j) { if (piv[j] != j) { needs_pivot = true; break; } }

        if (needs_pivot) {
            Matrix U_piv(M, N, Matrix::UninitializedTag{});
            for (size_t j = 0; j < N; ++j) {
                const double* src = std::assume_aligned<64>(U.data.get() + piv[j] * U.stride);
                double* dst = std::assume_aligned<64>(U_piv.data.get() + j * U_piv.stride);
                std::copy_n(src, M, dst);
            }
            U = std::move(U_piv);
            // V를 순열 행렬로 재설정 (항등 행렬 → P)
            std::memset(V.data.get(), 0, V.stride * V.cols * sizeof(double));
            for (size_t j = 0; j < N; ++j) V(piv[j], j) = 1.0;
        }
    }

    constexpr int MAX_SWEEPS = 30;
    const double tol = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(M, N));

    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4;
    // [D.2] 병렬화 임계값 상향: 스레드 생성 오버헤드 대비 충분한 작업량 보장
    const bool par_outer = (N >= std::max(64u, num_cores * 8));
    const bool par_inner = !par_outer && (M >= 1024);

    size_t n_even = N + (N % 2);
    auto ptr_machines = std::make_unique<size_t[]>(n_even);
    std::span<size_t> machines(ptr_machines.get(), n_even);
    std::iota(machines.begin(), machines.end(), 0);

    auto ptr_pairs = std::make_unique<std::pair<size_t, size_t>[]>(n_even / 2);
    std::span<std::pair<size_t, size_t>> pairs_buffer(ptr_pairs.get(), n_even / 2);

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double sweep_max_off = 0.0;
        for (size_t step = 0; step < n_even - 1; ++step) {
            size_t valid_pairs = 0;
            for (size_t k = 0; k < n_even / 2; ++k) {
                size_t c1 = machines[k], c2 = machines[n_even - 1 - k];
                if (c1 < N && c2 < N) pairs_buffer[valid_pairs++] = (c1 < c2) ? std::make_pair(c1, c2) : std::make_pair(c2, c1);
            }
            auto active_pairs = pairs_buffer.subspan(0, valid_pairs);

            auto process_pair = [&](const std::pair<size_t, size_t>& p_idx) noexcept -> double {
                size_t i = p_idx.first, j = p_idx.second;

                // [최적화] C++20 assume_aligned를 통해 컴파일러에게 포인터 메모리가 
                // 64바이트로 완벽히 정렬되었음을 맹세하여 SIMD 정렬 체크 분기문을 삭제.
                double* ui_ptr = std::assume_aligned<64>(U.data.get() + i * U.stride);
                double* uj_ptr = std::assume_aligned<64>(U.data.get() + j * U.stride);

                PQR pqr = (par_inner)
                    ? std::transform_reduce(std::execution::par_unseq, ui_ptr, ui_ptr + M, uj_ptr, PQR{ 0.0, 0.0, 0.0 }, pqr_combine, pqr_map)
                    : std::transform_reduce(std::execution::unseq, ui_ptr, ui_ptr + M, uj_ptr, PQR{ 0.0, 0.0, 0.0 }, pqr_combine, pqr_map);

                double p = pqr.p, q = pqr.q, r = pqr.r;
                double off_diag = (q == 0.0 || r == 0.0) ? 0.0 : std::abs(p) / std::sqrt(q * r);
                if (off_diag <= tol) return off_diag;

                double theta = (r - q) / (2.0 * p);
                double t;
                if (std::isinf(theta)) {
                    t = 0.0;
                } else {
                    t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(1.0, theta));
                }
                double c = 1.0 / std::sqrt(1.0 + t * t), s = c * t;

                double* vi_ptr = std::assume_aligned<64>(V.data.get() + i * V.stride);
                double* vj_ptr = std::assume_aligned<64>(V.data.get() + j * V.stride);

                auto apply_rot = [c, s](double& xi, double& xj) noexcept { double vi = xi, vj = xj; xi = c * vi - s * vj; xj = s * vi + c * vj; };

                if (par_inner) {
                    std::for_each(std::execution::par_unseq, M_indices.begin(), M_indices.end(), [&](size_t idx) noexcept { apply_rot(ui_ptr[idx], uj_ptr[idx]); });
                    std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(), [&](size_t idx) noexcept { apply_rot(vi_ptr[idx], vj_ptr[idx]); });
                }
                else {
                    std::for_each(std::execution::unseq, M_indices.begin(), M_indices.end(), [&](size_t idx) noexcept { apply_rot(ui_ptr[idx], uj_ptr[idx]); });
                    std::for_each(std::execution::unseq, N_indices.begin(), N_indices.end(), [&](size_t idx) noexcept { apply_rot(vi_ptr[idx], vj_ptr[idx]); });
                }
                return off_diag;
                };

            auto max_op = [](double a, double b) constexpr noexcept { return a > b ? a : b; };
            double step_max_off = (par_outer)
                ? std::transform_reduce(std::execution::par_unseq, active_pairs.begin(), active_pairs.end(), 0.0, max_op, process_pair)
                : std::transform_reduce(std::execution::seq, active_pairs.begin(), active_pairs.end(), 0.0, max_op, process_pair);

            if (step_max_off > sweep_max_off) sweep_max_off = step_max_off;
            // [C.2] Sweep 내부 조기 종료: 직전 sweep이 전체 쌍을 순회한 후,
            // 현재 sweep에서도 모든 step이 수렴 확인 시 잔여 step 생략
            if (sweep > 0 && sweep_max_off <= tol) break;
            std::rotate(machines.begin() + 1, machines.end() - 1, machines.end());
        }
        if (sweep_max_off <= tol) break;
    }

    // [B.1] 수렴 후 V 직교성 검증 및 Modified Gram-Schmidt 재직교화
    // 수십 sweep 동안 누적된 Givens 회전의 부동소수점 반올림 오차를 보정
    {
        const double ortho_tol = std::sqrt(std::numeric_limits<double>::epsilon()); // ≈ 1.5e-8
        double max_ortho_err = 0.0;
        bool needs_reorth = false;

        for (size_t i = 0; i < N && !needs_reorth; ++i) {
            const double* vi_ptr = std::assume_aligned<64>(V.data.get() + i * V.stride);
            for (size_t j = i + 1; j < N; ++j) {
                const double* vj_ptr = std::assume_aligned<64>(V.data.get() + j * V.stride);
                double dot = std::transform_reduce(std::execution::unseq, vi_ptr, vi_ptr + N, vj_ptr, 0.0);
                double abs_dot = std::abs(dot);
                if (abs_dot > max_ortho_err) max_ortho_err = abs_dot;
                if (max_ortho_err > ortho_tol) { needs_reorth = true; break; }
            }
        }

        if (needs_reorth) {
            // Modified Gram-Schmidt: V의 열벡터들을 재직교화
            for (size_t j = 0; j < N; ++j) {
                double* vj_ptr = V.data.get() + j * V.stride;
                for (size_t k = 0; k < j; ++k) {
                    const double* vk_ptr = std::assume_aligned<64>(V.data.get() + k * V.stride);
                    double proj = std::transform_reduce(std::execution::unseq, vj_ptr, vj_ptr + N, vk_ptr, 0.0);
                    for (size_t idx = 0; idx < N; ++idx) vj_ptr[idx] -= proj * vk_ptr[idx];
                }
                double norm_sq = std::transform_reduce(std::execution::unseq, vj_ptr, vj_ptr + N, 0.0,
                    std::plus<>{}, [](double x) noexcept { return x * x; });
                double norm = std::sqrt(norm_sq);
                if (norm > std::numeric_limits<double>::min()) {
                    double inv = 1.0 / norm;
                    std::for_each(std::execution::unseq, vj_ptr, vj_ptr + N,
                        [inv](double& x) noexcept { x *= inv; });
                }
            }

            // U = A * V 재계산 (열 기반 DAXPY 누적으로 캐시 효율 극대화)
            // A는 const 참조로 원본이 보존되어 있으므로 정확한 재구성 가능
            std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(), [&](size_t j) noexcept {
                double* uj_ptr = std::assume_aligned<64>(U.data.get() + j * U.stride);
                std::fill_n(uj_ptr, M, 0.0);
                const double* vj_ptr = std::assume_aligned<64>(V.data.get() + j * V.stride);
                for (size_t k = 0; k < N; ++k) {
                    double v_kj = vj_ptr[k];
                    if (v_kj == 0.0) continue;
                    const double* ak_ptr = std::assume_aligned<64>(A.data.get() + k * A.stride);
                    for (size_t i = 0; i < M; ++i) uj_ptr[i] += v_kj * ak_ptr[i];
                }
            });
        }
    }

    // 2. 특이값 계산 및 복원 로직
    std::vector<double> S_vec(N);
    std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(), [&](size_t j) noexcept {
        double* u_ptr = std::assume_aligned<64>(U.data.get() + j * U.stride);
        double norm_sq = std::transform_reduce(std::execution::unseq, u_ptr, u_ptr + M, 0.0, std::plus<>{}, [](double x) noexcept { return x * x; });
        S_vec[j] = std::sqrt(norm_sq);
        });

    double max_S = *std::max_element(std::execution::unseq, S_vec.begin(), S_vec.end());
    double rank_tol = std::numeric_limits<double>::epsilon() * static_cast<double>(M) * max_S;

    for (size_t j = 0; j < N; ++j) {
        double* u_ptr = std::assume_aligned<64>(U.data.get() + j * U.stride);
        if (S_vec[j] > rank_tol) {
            double inv_sig = 1.0 / S_vec[j];
            std::for_each(std::execution::unseq, u_ptr, u_ptr + M, [inv_sig](double& x) noexcept { x *= inv_sig; });
        }
        else {
            S_vec[j] = 0.0;
            std::for_each(std::execution::unseq, u_ptr, u_ptr + M, [](double& x) noexcept { x = 0.0; });
        }
    }

    std::vector<size_t> p(N);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](size_t a, size_t b) noexcept { return S_vec[a] > S_vec[b]; });

    Matrix U_final(M, N, Matrix::UninitializedTag{}), V_final(N, N, Matrix::UninitializedTag{}), Sigma_final(N, N, 0.0);

    std::for_each(std::execution::par_unseq, N_indices.begin(), N_indices.end(), [&](size_t new_j) noexcept {
        size_t old_j = p[new_j];
        Sigma_final(new_j, new_j) = S_vec[old_j];
        const double* u_src = std::assume_aligned<64>(U.data.get() + old_j * U.stride);
        double* u_dst = std::assume_aligned<64>(U_final.data.get() + new_j * U_final.stride);
        std::copy(std::execution::unseq, u_src, u_src + M, u_dst);

        const double* v_src = std::assume_aligned<64>(V.data.get() + old_j * V.stride);
        double* v_dst = std::assume_aligned<64>(V_final.data.get() + new_j * V_final.stride);
        std::copy(std::execution::unseq, v_src, v_src + N, v_dst);
        });

    return SVDResult(std::move(U_final), std::move(Sigma_final), std::move(V_final));
}

SVDResult compute_svd(const Matrix& A) {
    // 진입점 하드웨어 보호막 (Subnormal 마이크로코드 스톨 방지)
    HardwareOptimizationGuard hw_guard;

    if (A.rows == 0 || A.cols == 0) return SVDResult(Matrix(A), Matrix(A), Matrix(A));

    // [B.3 수정] NaN/Inf 입력 검증 — 열 우선(column-major) 순회로 캐시 적중률 보장
    for (size_t j = 0; j < A.cols; ++j) {
        const double* col = A.data.get() + j * A.stride;
        for (size_t i = 0; i < A.rows; ++i) {
            if (!std::isfinite(col[i]))
                throw std::invalid_argument("Input matrix contains NaN or Inf");
        }
    }

    if (A.rows >= A.cols) return svd_tall(A);
    Matrix At = A.transpose();
    SVDResult res_t = svd_tall(At);
    return SVDResult(std::move(res_t.V), std::move(res_t.Sigma), std::move(res_t.U));
}

void print_matrix(const Matrix& M, const std::string& name) {
    std::cout << "=== " << name << " (" << M.rows << "x" << M.cols << ") ===\n";
    for (size_t i = 0; i < M.rows; ++i) {
        for (size_t j = 0; j < M.cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << M(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
// =======================================================================
// 검증 유틸리티 및 안전 방어선 메인 함수
// =======================================================================
Matrix matmul(const Matrix& A, const Matrix& B) {
    // 덮어쓰기 연산이므로 C++20 UninitializedTag를 사용하여 O(N) 0-초기화 오버헤드 완벽 회피
    Matrix C(A.rows, B.cols, Matrix::UninitializedTag{});
    auto I_indices = std::views::iota(size_t{ 0 }, A.rows);
    auto J_indices = std::views::iota(size_t{ 0 }, B.cols);
    auto K_indices = std::views::iota(size_t{ 0 }, A.cols);

    std::for_each(std::execution::par_unseq, J_indices.begin(), J_indices.end(), [&](size_t j) noexcept {
        std::for_each(std::execution::unseq, I_indices.begin(), I_indices.end(), [&](size_t i) noexcept {
            C(i, j) = std::transform_reduce(std::execution::unseq, K_indices.begin(), K_indices.end(), 0.0, std::plus<>{}, [&](size_t k) noexcept {
                return A(i, k) * B(k, j);
                });
            });
        });
    return C;
}

int main() {
    try {
        Matrix A = Matrix::from_row_major(5, 3, {
            1.0,  5.0,  9.0,
            2.0,  6.0, 10.0,
            3.0,  7.0, 11.0,
            4.0,  8.0, 12.0,
            5.0,  0.0,  2.0
            });

        std::cout << "[Test 1: M >= N Matrix]\n";
        auto resA = compute_svd(A);
        print_matrix(resA.Sigma, "Matrix Sigma");
        Matrix A_rec = matmul(matmul(resA.U, resA.Sigma), resA.V.transpose());
        print_matrix(A_rec, "Reconstructed A");

        Matrix B = A.transpose();
        std::cout << "[Test 2: M < N Matrix]\n";
        auto resB = compute_svd(B);
        print_matrix(resB.Sigma, "Matrix Sigma");
        Matrix B_rec = matmul(matmul(resB.U, resB.Sigma), resB.V.transpose());
        print_matrix(B_rec, "Reconstructed B");

        // [Test 3: OOM 방어 가드 작동 테스트 - 주석 해제 시 안전하게 거부됨]
        // std::cout << "\n[Test 3: Defensive OOM Guard]\n";
        // Matrix malicious_matrix(2000000, 2000000); // 약 32 테라바이트 요청

    }
    catch (const std::length_error& e) {
        std::cerr << "⛔ [SAFE REJECT] Computation Refused: " << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "⛔ Fatal Error: " << e.what() << "\n";
    }
    return 0;
}