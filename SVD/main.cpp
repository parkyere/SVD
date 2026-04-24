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
#include <new>         // std::align_val_t (C++17 표준 — assume_aligned는 C++20)
#include <functional>  // std::logical_or 등
#include <cstring>     // for std::memset
#include <cassert>     // for assert()
#include <string>      // for std::to_string in error messages

// =======================================================================
// [1] 64-Byte (Cache-Line) 정렬 메모리 할당기 + 크기 안전 가드
// C++20 표준 정렬 operator new/delete 사용 — 플랫폼 조건 분기 완전 제거
// =======================================================================

// 단일 행렬 1개 할당의 절대 상한. 악의적/실수성 거대 입력에 대한 1차 방어선.
// 외부에서 -DSVD_MAX_MATRIX_BYTES=... 로 컴파일 시 재정의 가능.
#ifndef SVD_MAX_MATRIX_BYTES
constexpr size_t SVD_MAX_MATRIX_BYTES = (8ULL << 30); // 8 GiB
#endif
constexpr size_t SVD_MAX_MATRIX_DOUBLES = SVD_MAX_MATRIX_BYTES / sizeof(double);

// 검증된 곱셈: size_t 오버플로 발생 시 length_error로 명시적 거부.
inline size_t safe_mul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::length_error("matrix dimension multiplication overflow");
    return a * b;
}

// 요소 수가 안전 한계 이내인지 확인.
inline void check_element_count(size_t count) {
    if (count > SVD_MAX_MATRIX_DOUBLES) {
        throw std::length_error(
            "matrix allocation exceeds safety limit ("
            + std::to_string(SVD_MAX_MATRIX_BYTES >> 30) + " GiB; requested "
            + std::to_string(count * sizeof(double)) + " bytes)");
    }
}

struct AlignedFree {
    void operator()(void* p) const noexcept {
        ::operator delete(p, std::align_val_t{ 64 });
    }
};
using AlignedDoublePtr = std::unique_ptr<double[], AlignedFree>;

inline AlignedDoublePtr allocate_aligned_uninitialized(size_t count) {
    if (count == 0) return nullptr;
    check_element_count(count);                               // 1차 가드 (안전 상한)
    size_t bytes = safe_mul(count, sizeof(double));           // 2차 가드 (오버플로)
    size_t alloc_bytes = (bytes + 63) & ~size_t{ 63 };        // 64바이트 배수로 올림
    if (alloc_bytes < bytes)                                  // 3차 가드 (rounding 오버플로)
        throw std::length_error("matrix allocation byte rounding overflow");
    void* ptr = ::operator new(alloc_bytes, std::align_val_t{ 64 });
    return AlignedDoublePtr(static_cast<double*>(ptr));
}

struct PQR { double p, q, r; };
constexpr auto pqr_combine = [](const PQR& a, const PQR& b) constexpr noexcept -> PQR { return { a.p + b.p, a.q + b.q, a.r + b.r }; };
constexpr auto pqr_map = [](double vi, double vj) constexpr noexcept -> PQR { return { vi * vj, vi * vi, vj * vj }; };

// =======================================================================
// [2] HPC Architecture 적용 Matrix (Padding & Stride 도입)
// =======================================================================
class Matrix {
public:
    size_t rows, cols;
    size_t stride; // 메모리상의 실제 행 길이 (캐시 뱅크 충돌 방지용 Padding 포함, LAPACK의 'lda')
    AlignedDoublePtr data;

    struct UninitializedTag {};

    // [최적화] 행(Row) 길이를 강제로 64-Byte(8 double) 배수로 맞춰 루프 테일(Tail)을 삭제하고,
    // 2의 거듭제곱 사이즈로 인한 L1 캐시 뱅크 충돌(Bank Conflict)을 원천 회피합니다.
    // 오버플로 가드: r이 SIZE_MAX-15 이상이면 length_error를 던져 silent wrap을 차단.
    static size_t calculate_stride(size_t r) {
        if (r == 0) return 0;
        if (r > std::numeric_limits<size_t>::max() - 15)
            throw std::length_error("row dimension too large for stride padding");
        size_t s = (r + 7) & ~size_t{ 7 };
        if (s >= 64 && (s & (s - 1)) == 0) s += 8; // 2의 거듭제곱일 경우 패딩 추가
        return s;
    }

    Matrix(size_t r, size_t c, double val = 0.0) : rows(r), cols(c), stride(calculate_stride(r)) {
        const size_t total = safe_mul(stride, c);
        if (total > 0) {
            data = allocate_aligned_uninitialized(total);
            if (val != 0.0) {
                double* ptr = std::assume_aligned<64>(data.get());
                std::fill_n(std::execution::par_unseq, ptr, total, val);
            }
            else {
                std::memset(data.get(), 0, safe_mul(total, sizeof(double)));
            }
        }
    }

    Matrix(size_t r, size_t c, UninitializedTag) : rows(r), cols(c), stride(calculate_stride(r)) {
        const size_t total = safe_mul(stride, c);
        if (total > 0) data = allocate_aligned_uninitialized(total);
    }

    Matrix() noexcept : rows(0), cols(0), stride(0), data(nullptr) {}

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), stride(other.stride) {
        const size_t total = safe_mul(stride, cols);
        if (total > 0) {
            data = allocate_aligned_uninitialized(total);
            const double* src = std::assume_aligned<64>(other.data.get());
            double* dst = std::assume_aligned<64>(data.get());
            std::copy(std::execution::par_unseq, src, src + total, dst);
        }
    }

    Matrix(Matrix&& other) noexcept : rows(std::exchange(other.rows, 0)), cols(std::exchange(other.cols, 0)), stride(std::exchange(other.stride, 0)), data(std::move(other.data)) {}

    Matrix& operator=(Matrix&& other) noexcept {
        rows = std::exchange(other.rows, 0); cols = std::exchange(other.cols, 0);
        stride = std::exchange(other.stride, 0); data = std::move(other.data);
        return *this;
    }

    Matrix& operator=(const Matrix& other) { if (this != &other) { Matrix tmp(other); *this = std::move(tmp); } return *this; }

    // 데이터 접근 시 rows가 아닌 stride를 곱하여 캐시 라인 오프셋을 적용.
    // operator()는 hot path이므로 디버그에서만 검사 (NDEBUG에서 무비용).
    double& operator()(size_t i, size_t j) noexcept {
        assert(i < rows && j < cols && "Matrix::operator() index out of range");
        return data.get()[j * stride + i];
    }
    const double& operator()(size_t i, size_t j) const noexcept {
        assert(i < rows && j < cols && "Matrix::operator() index out of range");
        return data.get()[j * stride + i];
    }

    // 외부 입력을 다룰 때 사용할 안전 접근자: 릴리스에서도 범위 검사 후 throw.
    double& at(size_t i, size_t j) {
        if (i >= rows || j >= cols)
            throw std::out_of_range("Matrix::at index out of range");
        return data.get()[j * stride + i];
    }
    const double& at(size_t i, size_t j) const {
        if (i >= rows || j >= cols)
            throw std::out_of_range("Matrix::at index out of range");
        return data.get()[j * stride + i];
    }

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
        // 오버플로 안전 검증: r * c가 size_t를 wrap하면 flat.size() 비교가 무력화되어
        // OOB 읽기가 발생할 수 있다. safe_mul로 명시적 거부.
        const size_t needed = safe_mul(r, c);
        if (flat.size() < needed)
            throw std::invalid_argument("flat vector size insufficient for matrix dimensions");
        Matrix M(r, c, 0.0); // 패딩 영역 0 초기화를 위해 기본 생성자 사용
        for (size_t i = 0; i < r; ++i) for (size_t j = 0; j < c; ++j) M(i, j) = flat[i * c + j];
        return M;
    }
};

struct SVDResult { Matrix U, Sigma, V; SVDResult(Matrix&& u, Matrix&& s, Matrix&& v) noexcept : U(std::move(u)), Sigma(std::move(s)), V(std::move(v)) {} };

// 전방 선언: QRCP 전처리 후 Q·U_R 합성에 사용. 정의는 파일 후반부.
Matrix matmul(const Matrix& A, const Matrix& B);

// =======================================================================
// [3] 현업 투입 레벨 Hestenes Jacobi SVD Engine (assume_aligned 극한 활용)
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

        // [Threshold Jacobi] 초반 sweep에선 작은 off-diag pair에 대한 회전을 skip해
        // 무의미한 작업을 줄인다. sweep이 진행될수록 threshold는 진짜 수렴 임계값 tol에
        // 단조감소 수렴 — 후반 sweep는 classical Hestenes와 완전 동일하게 동작.
        // 측정값(off_diag) 자체는 항상 반환되어 sweep_max_off / 수렴 검사는 honest.
        const double sweep_threshold = std::max(tol, std::pow(0.1, sweep + 1));

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
                // q*r 직접 곱은 underflow(→0)나 overflow(→inf)에 취약하므로
                // sqrt를 분리 적용해 안정 범위를 확장.
                const double sq = std::sqrt(q), sr = std::sqrt(r);
                double off_diag = (sq == 0.0 || sr == 0.0) ? 0.0 : std::abs(p) / (sq * sr);
                // 회전은 sweep_threshold 이상일 때만; off_diag 자체는 늘 반환해 정직한 수렴 측정 보장.
                if (off_diag <= sweep_threshold) return off_diag;

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
            // (기존의 "sweep > 0 && sweep_max_off <= tol" 중간 break는 부분 step만 보고
            //  남은 pair의 off-diag를 모르는 채 조기 종료하는 unsound 최적화였으므로 제거.
            //  수렴 판정은 반드시 한 sweep을 완전히 끝낸 뒤에만 수행한다.)
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
            // Modified Gram-Schmidt: V의 열벡터들을 재직교화.
            // "Twice is enough" (Giraud, Langou, Rozložník 2005): 1회 MGS는 조건수가 큰 경우
            // 직교성을 epsilon 수준까지 회복하지 못할 수 있으므로 두 번 적용한다.
            for (int pass = 0; pass < 2; ++pass) {
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

    const double max_S = *std::max_element(std::execution::unseq, S_vec.begin(), S_vec.end());
    // 절대 하한 추가: max_S가 subnormal에 가깝거나 0인 병적 입력에서 0으로 떨어지지 않도록 보호.
    const double rank_tol = std::max(
        std::numeric_limits<double>::epsilon() * static_cast<double>(M) * max_S,
        std::numeric_limits<double>::min());

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

// =======================================================================
// [4] 외부 진입점: Thin / Full SVD
// 모드별 반환 차원 (입력 A: M×N) — 두 경우 모두 A = U Σ Vᵀ 성립:
//   Thin (default, 빠르고 메모리 절약):
//     - M >= N: U (M×N, 정규직교 열),  Σ (N×N 대각),  V (N×N 직교)
//     - M <  N: U (M×M 직교),          Σ (M×M 대각),  V (N×M 정규직교 열)
//   Full (정사각 직교 U·V):
//     - 항상   : U (M×M 직교),          Σ (M×N 대각 패딩), V (N×N 직교)
// =======================================================================
enum class SVDMode { Thin, Full };

// 정규직교 열 k개를 가진 Q (rows×k, k <= rows)를 rows×rows 정사각 직교 행렬로 확장.
// 표준 기저 e_0…e_{rows-1}을 차례로 시도하고, 각 후보를 기존 열들에 대해 MGS twice로
// 직교화한 뒤 노름이 충분하면 새 열로 채택. e_i들과 Q의 열은 함께 R^rows를 span하므로
// 이론적으로 항상 rows-k개의 유효 후보를 찾을 수 있다 (Householder QR 대비 단순 명시 구현).
Matrix extend_to_orthogonal(const Matrix& Q) {
    const size_t rows = Q.rows, k = Q.cols;
    if (k > rows)
        throw std::invalid_argument("extend_to_orthogonal: cols > rows");
    if (k == rows) return Q; // 이미 정사각

    Matrix Q_full(rows, rows, 0.0);
    for (size_t j = 0; j < k; ++j) {
        const double* src = Q.data.get() + j * Q.stride;
        double* dst = Q_full.data.get() + j * Q_full.stride;
        std::copy_n(src, rows, dst);
    }

    const double accept_tol = std::sqrt(std::numeric_limits<double>::epsilon());
    std::vector<double> cand(rows);

    size_t added = k;
    for (size_t e = 0; e < rows && added < rows; ++e) {
        std::fill(cand.begin(), cand.end(), 0.0);
        cand[e] = 1.0;

        // MGS twice — 1회로는 직교성 회복이 불충분할 수 있음
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t j = 0; j < added; ++j) {
                const double* qj = Q_full.data.get() + j * Q_full.stride;
                double dot = std::transform_reduce(std::execution::unseq,
                    cand.begin(), cand.end(), qj, 0.0);
                for (size_t i = 0; i < rows; ++i) cand[i] -= dot * qj[i];
            }
        }

        const double norm_sq = std::transform_reduce(std::execution::unseq,
            cand.begin(), cand.end(), 0.0, std::plus<>{},
            [](double x) noexcept { return x * x; });
        const double norm = std::sqrt(norm_sq);

        if (norm > accept_tol) {
            const double inv = 1.0 / norm;
            double* dst = Q_full.data.get() + added * Q_full.stride;
            for (size_t i = 0; i < rows; ++i) dst[i] = cand[i] * inv;
            ++added;
        }
    }

    if (added != rows) {
        // 표준 기저로 보충이 안 되는 병적 케이스 (이론상 발생 안 함)
        throw std::runtime_error("orthogonal extension failed: insufficient basis vectors");
    }
    return Q_full;
}

// k = min(target_M, target_N) 개의 특이값을 target_M×target_N Σ로 패딩.
inline Matrix make_padded_sigma(size_t target_M, size_t target_N, const Matrix& Sigma_diag) {
    Matrix S(target_M, target_N, 0.0);
    const size_t k = std::min({ target_M, target_N, Sigma_diag.rows, Sigma_diag.cols });
    for (size_t i = 0; i < k; ++i) S(i, i) = Sigma_diag(i, i);
    return S;
}

// =======================================================================
// [5] Householder reflector + Businger-Golub QRCP (Drmač-Veselić 전처리)
// =======================================================================

// Householder reflector H = I - tau · v · vᵀ.
// 입력 길이 n 벡터 x를 e_0 방향 spike로 반사: H·x = alpha·e_0 (|alpha| = ||x||₂).
// LAPACK 관행에 따라 v[0] = 1 정규화 (essential part는 v[1..n-1]).
// 부호는 cancellation 회피: alpha = -copysign(||x||, x[0]).
struct HouseholderResult {
    std::vector<double> v; // length n; v[0] == 1
    double tau;            // 2 / (vᵀv)
    double alpha;          // 반사 후 spike 값 (= ±||x||)
};

inline HouseholderResult make_householder(const double* x, size_t n) {
    if (n == 0) return { {}, 0.0, 0.0 };
    if (n == 1) return { { 1.0 }, 0.0, x[0] };

    const double xnorm_sq_rest = std::transform_reduce(std::execution::unseq,
        x + 1, x + n, 0.0, std::plus<>{},
        [](double v) noexcept { return v * v; });

    if (xnorm_sq_rest == 0.0) {
        // 이미 e_0 방향 — 반사 불필요
        return { std::vector<double>(n, 0.0), 0.0, x[0] };
    }

    const double xnorm = std::sqrt(x[0] * x[0] + xnorm_sq_rest);
    const double alpha = -std::copysign(xnorm, x[0]); // 부호: x[0]과 반대로 → 큰 v0
    const double v0 = x[0] - alpha;
    const double inv_v0 = 1.0 / v0;

    std::vector<double> v(n);
    v[0] = 1.0;
    for (size_t i = 1; i < n; ++i) v[i] = x[i] * inv_v0;

    // tau = 2 / (vᵀv)
    double vtv = 1.0;
    for (size_t i = 1; i < n; ++i) vtv += v[i] * v[i];
    const double tau = 2.0 / vtv;

    return { std::move(v), tau, alpha };
}

// y ← H·y (in-place). y와 H.v의 길이가 같다고 가정.
inline void apply_householder(const HouseholderResult& h, double* y) noexcept {
    if (h.tau == 0.0) return;
    const size_t n = h.v.size();
    double dot = 0.0;
    for (size_t i = 0; i < n; ++i) dot += h.v[i] * y[i];
    const double scale = h.tau * dot;
    for (size_t i = 0; i < n; ++i) y[i] -= scale * h.v[i];
}

// =======================================================================
// Implicit Q storage: Householder reflector v[1..]는 W의 sub-diagonal에,
// τ는 별도 taus[]에 저장 — 별도 Q 행렬과 reflectors[] 컨테이너 모두 제거.
// 메모리 사용량: 기존 ~3·MN → ~MN으로 약 3배 절감.
// =======================================================================

// QRCP (column pivoting) 결과 — implicit Q.
struct QRCPCompact {
    Matrix W;                 // M × N: upper triangle = R, sub-diagonal = essential v[1..]
    std::vector<double> taus; // length N
    std::vector<size_t> perm; // perm[j] = R의 j번째 열에 들어간 원본 A 열 index
};

// Plain QR (no column pivoting) 결과 — implicit Q. 2단계 QRCP의 stage 2에서 사용.
struct QRCompact {
    Matrix W;                 // M × N: upper triangle = R, sub-diagonal = essential v[1..]
    std::vector<double> taus; // length N
};

// W의 upper triangle에서 N×N 상삼각 R을 복사 추출.
inline Matrix extract_R_from_compact(const Matrix& W) {
    const size_t N = W.cols;
    Matrix R(N, N, 0.0);
    for (size_t j = 0; j < N; ++j) {
        const double* col = W.data.get() + j * W.stride;
        for (size_t i = 0; i <= j; ++i) R(i, j) = col[i];
    }
    return R;
}

// target ← Q · target,  여기서 Q = H_0·H_1·…·H_{N_ref-1},  H_j는 W의 column j의
// sub-diagonal에 v[1..] 압축저장 (v[0] = 1 implicit), τ는 taus[j].
// target은 W.rows × K. 작용 순서는 H_{N_ref-1}, …, H_0 (역순).
inline void apply_Q_left_inplace(const Matrix& W, const std::vector<double>& taus,
    Matrix& target) {
    const size_t M = W.rows, N_ref = W.cols, K = target.cols;
    if (target.rows != M)
        throw std::invalid_argument("apply_Q_left_inplace: target rows != W.rows");

    for (size_t step = N_ref; step > 0; --step) {
        const size_t j = step - 1;
        const double tau_j = taus[j];
        if (tau_j == 0.0) continue;
        const double* w_col_j = W.data.get() + j * W.stride;
        const size_t v_len = M - j;

        auto cols_iota = std::views::iota(size_t{ 0 }, K);
        std::for_each(std::execution::par_unseq, cols_iota.begin(), cols_iota.end(),
            [&, j_local = j, v_len_local = v_len, tau_local = tau_j](size_t k) noexcept {
                double* y = target.data.get() + k * target.stride + j_local;
                // dot = v[0]·y[0] + Σ v[i]·y[i],  v[0] = 1
                double dot = y[0];
                for (size_t i = 1; i < v_len_local; ++i) dot += w_col_j[j_local + i] * y[i];
                const double scale = tau_local * dot;
                y[0] -= scale; // v[0] = 1
                for (size_t i = 1; i < v_len_local; ++i) y[i] -= scale * w_col_j[j_local + i];
            });
    }
}

// Q (implicit, M × N_ref)를 N_ref × N_ref U_R에 적용 → M × N_ref 결과.
// U_R을 M × N_ref로 임베딩 (top N_ref rows = U_R, bottom (M - N_ref) rows = 0) 후 in-place 적용.
inline Matrix apply_Q_to_thin_target(const Matrix& W, const std::vector<double>& taus,
    const Matrix& U_R) {
    const size_t M = W.rows, N = W.cols;
    if (U_R.rows != N || U_R.cols != N)
        throw std::invalid_argument("apply_Q_to_thin_target: U_R must be N × N");

    Matrix result(M, N, 0.0);
    // U_R을 result의 상위 N행에 복사
    for (size_t j = 0; j < N; ++j) {
        const double* src = U_R.data.get() + j * U_R.stride;
        double* dst = result.data.get() + j * result.stride;
        std::copy_n(src, N, dst);
    }
    apply_Q_left_inplace(W, taus, result);
    return result;
}

// Businger-Golub QR with column pivoting (implicit Q storage).
// Pre: A.rows >= A.cols.
// 만족 등식: A·P = Q·R,  여기서 |R(0,0)| ≥ |R(1,1)| ≥ … (단조감소 보장).
QRCPCompact businger_golub_qrcp(const Matrix& A) {
    if (A.rows < A.cols)
        throw std::invalid_argument("businger_golub_qrcp requires rows >= cols");

    const size_t M = A.rows, N = A.cols;
    Matrix W = A;

    std::vector<size_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<double> taus(N, 0.0);

    // 초기 열 norm²
    std::vector<double> col_norms_sq(N), original_norms_sq(N);
    {
        auto cols_iota = std::views::iota(size_t{ 0 }, N);
        std::for_each(std::execution::par_unseq, cols_iota.begin(), cols_iota.end(),
            [&](size_t j) noexcept {
                const double* col = W.data.get() + j * W.stride;
                const double s = std::transform_reduce(std::execution::unseq,
                    col, col + M, 0.0, std::plus<>{},
                    [](double x) noexcept { return x * x; });
                col_norms_sq[j] = s;
                original_norms_sq[j] = s;
            });
    }

    for (size_t j = 0; j < N; ++j) {
        // 1. Pivot 선택
        size_t pivot = j;
        double max_norm_sq = col_norms_sq[j];
        for (size_t k = j + 1; k < N; ++k) {
            if (col_norms_sq[k] > max_norm_sq) {
                max_norm_sq = col_norms_sq[k];
                pivot = k;
            }
        }
        if (pivot != j) {
            double* col_j = W.data.get() + j * W.stride;
            double* col_p = W.data.get() + pivot * W.stride;
            std::swap_ranges(col_j, col_j + M, col_p);
            std::swap(col_norms_sq[j], col_norms_sq[pivot]);
            std::swap(original_norms_sq[j], original_norms_sq[pivot]);
            std::swap(perm[j], perm[pivot]);
        }

        // 2. Householder + W에 압축 저장
        double* col_j = W.data.get() + j * W.stride;
        HouseholderResult h = make_householder(col_j + j, M - j);
        taus[j] = h.tau;
        col_j[j] = h.alpha;                                  // R 대각
        for (size_t i = 1; i < M - j; ++i)
            col_j[j + i] = h.v[i];                           // v[1..] sub-diagonal에 저장

        // 3. 남은 열들에 H_j 적용 + norm downdate
        if (j + 1 < N) {
            auto rest = std::views::iota(j + 1, N);
            std::for_each(std::execution::par_unseq, rest.begin(), rest.end(),
                [&, j_local = j](size_t k) noexcept {
                    double* col_k = W.data.get() + k * W.stride;
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

// Plain Householder QR (no column pivoting), implicit Q storage.
// 2단계 QRCP의 stage 2에서 R1ᵀ에 적용. 입력은 N × N square 가정 (일반화 가능).
QRCompact qr_no_pivot(const Matrix& A) {
    if (A.rows < A.cols)
        throw std::invalid_argument("qr_no_pivot requires rows >= cols");

    const size_t M = A.rows, N = A.cols;
    Matrix W = A;
    std::vector<double> taus(N, 0.0);

    for (size_t j = 0; j < N; ++j) {
        double* col_j = W.data.get() + j * W.stride;
        HouseholderResult h = make_householder(col_j + j, M - j);
        taus[j] = h.tau;
        col_j[j] = h.alpha;
        for (size_t i = 1; i < M - j; ++i) col_j[j + i] = h.v[i];

        if (j + 1 < N) {
            auto rest = std::views::iota(j + 1, N);
            std::for_each(std::execution::par_unseq, rest.begin(), rest.end(),
                [&, j_local = j](size_t k) noexcept {
                    double* col_k = W.data.get() + k * W.stride;
                    apply_householder(h, col_k + j_local);
                });
        }
    }
    return { std::move(W), std::move(taus) };
}

// V_out[perm[l], k] = V_in[l, k]  (즉 V_out = P · V_in, P는 순열행렬)
inline Matrix permute_rows_square(const std::vector<size_t>& perm, const Matrix& V_in) {
    const size_t N = V_in.rows;
    if (V_in.cols != N || perm.size() != N)
        throw std::invalid_argument("permute_rows_square: dimension mismatch");

    Matrix V_out(N, N, Matrix::UninitializedTag{});
    auto cols_iota = std::views::iota(size_t{ 0 }, N);
    std::for_each(std::execution::par_unseq, cols_iota.begin(), cols_iota.end(),
        [&](size_t k) noexcept {
            const double* in_col = V_in.data.get() + k * V_in.stride;
            double* out_col = V_out.data.get() + k * V_out.stride;
            for (size_t l = 0; l < N; ++l) out_col[perm[l]] = in_col[l];
        });
    return V_out;
}

// 2단계 QRCP 전처리 (Drmač-Veselić §3.5):
//   Stage 1: X·P1 = Q1·R1  (column pivoting QRCP)
//   Stage 2: R1ᵀ = Q2·R2   (plain QR, no pivoting — R1ᵀ는 이미 잘 sorted)
//   Jacobi : R2 = U_R2·Σ·V_R2ᵀ
//   합성   : X = (Q1·V_R2)·Σ·(P1·Q2·U_R2)ᵀ
// 두 번의 QR로 작은 특이값의 상대 정확도가 한 단계 더 향상 (ill-conditioned 케이스).
SVDResult thin_svd_with_qrcp(const Matrix& X) {
    const size_t M = X.rows, N = X.cols;
    if (N <= 1) return svd_tall(X);

    // ── Stage 1: QRCP on X ────────────────────────────────────────────
    QRCPCompact qr1 = businger_golub_qrcp(X);
    Matrix R1 = extract_R_from_compact(qr1.W);   // N × N upper triangular

    // ── Stage 2: plain QR on R1ᵀ ─────────────────────────────────────
    Matrix R1_T = R1.transpose();                // N × N lower triangular
    QRCompact qr2 = qr_no_pivot(R1_T);
    Matrix R2 = extract_R_from_compact(qr2.W);   // N × N upper triangular

    // ── Jacobi SVD on R2 ─────────────────────────────────────────────
    SVDResult svd_R2 = svd_tall(R2);             // R2 = U_R2 · Σ · V_R2ᵀ

    // ── 합성 ─────────────────────────────────────────────────────────
    // R1 = (R1ᵀ)ᵀ = (Q2·R2)ᵀ = R2ᵀ·Q2ᵀ = (V_R2·Σ·U_R2ᵀ)·Q2ᵀ
    //    → R1의 SVD: U_R1 = V_R2,  V_R1 = Q2·U_R2
    // X = Q1·R1·P1ᵀ = (Q1·V_R2)·Σ·(P1·Q2·U_R2)ᵀ

    // U_X = Q1 · V_R2 — Q1 implicit, V_R2 (N×N) → M×N 결과
    Matrix U_X = apply_Q_to_thin_target(qr1.W, qr1.taus, svd_R2.V);

    // V_X = P1 · (Q2 · U_R2) — Q2 implicit, U_R2 (N×N) → N×N, 그 후 행 순열
    Matrix Q2_U_R2 = svd_R2.U; // copy
    apply_Q_left_inplace(qr2.W, qr2.taus, Q2_U_R2);
    Matrix V_X = permute_rows_square(qr1.perm, Q2_U_R2);

    return SVDResult(std::move(U_X), std::move(svd_R2.Sigma), std::move(V_X));
}

SVDResult compute_svd(const Matrix& A, SVDMode mode = SVDMode::Thin) {
    if (A.rows == 0 || A.cols == 0) return SVDResult(Matrix(A), Matrix(A), Matrix(A));

    // NaN/Inf 입력 검증 — 열 단위로 분할해 par_unseq로 병렬 검사.
    auto col_indices = std::views::iota(size_t{ 0 }, A.cols);
    const bool has_invalid = std::any_of(std::execution::par_unseq,
        col_indices.begin(), col_indices.end(),
        [&](size_t j) noexcept {
            const double* col = A.data.get() + j * A.stride;
            return std::any_of(std::execution::unseq, col, col + A.rows,
                [](double x) noexcept { return !std::isfinite(x); });
        });
    if (has_invalid)
        throw std::invalid_argument("Input matrix contains NaN or Inf");

    const size_t M = A.rows, N = A.cols;

    // 1단계: QRCP 전처리 + Jacobi (Drmač-Veselić). M < N이면 transpose해서 동일 경로 사용.
    SVDResult thin = (M >= N)
        ? thin_svd_with_qrcp(A)
        : [&]() {
            Matrix At = A.transpose();
            SVDResult rt = thin_svd_with_qrcp(At);
            // A = (Aᵀ)ᵀ = (Uₜ Σₜ Vₜᵀ)ᵀ = Vₜ Σₜ Uₜᵀ  →  U=Vₜ, Σ=Σₜ, V=Uₜ
            return SVDResult(std::move(rt.V), std::move(rt.Sigma), std::move(rt.U));
        }();

    if (mode == SVDMode::Thin) return thin;

    // 2단계: Full 모드에서 누락된 정사각 직교 보강
    // M >= N: thin.U는 M×N → M×M으로 확장.  V (N×N)는 이미 정사각.
    // M <  N: thin.V는 N×M → N×N으로 확장.  U (M×M)는 이미 정사각.
    // M == N: thin == full.
    if (M > N) {
        Matrix U_full = extend_to_orthogonal(thin.U);
        Matrix Sigma_full = make_padded_sigma(M, N, thin.Sigma);
        return SVDResult(std::move(U_full), std::move(Sigma_full), std::move(thin.V));
    }
    else if (M < N) {
        Matrix V_full = extend_to_orthogonal(thin.V);
        Matrix Sigma_full = make_padded_sigma(M, N, thin.Sigma);
        return SVDResult(std::move(thin.U), std::move(Sigma_full), std::move(V_full));
    }
    return thin; // M == N
}

void print_matrix(const Matrix& M, const std::string& name) {
    constexpr size_t MAX_PRINT_DIM = 16; // 콘솔 출력 폭주 방지
    std::cout << "=== " << name << " (" << M.rows << "x" << M.cols << ") ===\n";
    if (M.rows > MAX_PRINT_DIM || M.cols > MAX_PRINT_DIM) {
        std::cout << "(too large to print: limit "
            << MAX_PRINT_DIM << "x" << MAX_PRINT_DIM << ")\n\n";
        return;
    }
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

        std::cout << "[Test 1: M >= N Matrix, Thin SVD]\n";
        auto resA = compute_svd(A);
        print_matrix(resA.Sigma, "Matrix Sigma (Thin)");
        std::cout << "U dim: " << resA.U.rows << "x" << resA.U.cols
            << ", V dim: " << resA.V.rows << "x" << resA.V.cols << "\n\n";
        Matrix A_rec = matmul(matmul(resA.U, resA.Sigma), resA.V.transpose());
        print_matrix(A_rec, "Reconstructed A");

        Matrix B = A.transpose();
        std::cout << "[Test 2: M < N Matrix, Thin SVD]\n";
        auto resB = compute_svd(B);
        print_matrix(resB.Sigma, "Matrix Sigma (Thin)");
        std::cout << "U dim: " << resB.U.rows << "x" << resB.U.cols
            << ", V dim: " << resB.V.rows << "x" << resB.V.cols << "\n\n";
        Matrix B_rec = matmul(matmul(resB.U, resB.Sigma), resB.V.transpose());
        print_matrix(B_rec, "Reconstructed B");

        std::cout << "[Test 3: M >= N Matrix, Full SVD — U는 5x5 정사각]\n";
        auto resA_full = compute_svd(A, SVDMode::Full);
        print_matrix(resA_full.Sigma, "Matrix Sigma (Full, 5x3)");
        std::cout << "U dim: " << resA_full.U.rows << "x" << resA_full.U.cols
            << ", V dim: " << resA_full.V.rows << "x" << resA_full.V.cols << "\n\n";
        Matrix A_rec_full = matmul(matmul(resA_full.U, resA_full.Sigma), resA_full.V.transpose());
        print_matrix(A_rec_full, "Reconstructed A (Full)");

        std::cout << "[Test 4: M < N Matrix, Full SVD — V는 5x5 정사각]\n";
        auto resB_full = compute_svd(B, SVDMode::Full);
        print_matrix(resB_full.Sigma, "Matrix Sigma (Full, 3x5)");
        std::cout << "U dim: " << resB_full.U.rows << "x" << resB_full.U.cols
            << ", V dim: " << resB_full.V.rows << "x" << resB_full.V.cols << "\n\n";
        Matrix B_rec_full = matmul(matmul(resB_full.U, resB_full.Sigma), resB_full.V.transpose());
        print_matrix(B_rec_full, "Reconstructed B (Full)");

        // [Test 5: OOM 방어 가드 작동 테스트 - 주석 해제 시 안전하게 거부됨]
        // std::cout << "\n[Test 5: Defensive OOM Guard]\n";
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