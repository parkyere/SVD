#pragma once
//
// 내부: Householder reflector 기본 연산 (LAPACK 관행 v[0]=1 정규화).
//

#include <cstddef>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <execution>
#include <utility>

namespace svd::detail {

// H = I - tau · v · vᵀ.  H · x = alpha · e_0,  |alpha| = ||x||₂.
// alpha 부호는 cancellation 회피를 위해 -copysign(||x||, x[0]).
struct HouseholderResult {
    std::vector<double> v; // length n; v[0] == 1
    double tau;            // 2 / (vᵀv)
    double alpha;          // 반사 후 spike 값 (= ±||x||)
};

inline HouseholderResult make_householder(const double* x, std::size_t n) {
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

// y ← H · y (in-place).  y와 h.v의 length가 같다고 가정.
inline void apply_householder(const HouseholderResult& h, double* y) noexcept {
    if (h.tau == 0.0) return;
    const std::size_t n = h.v.size();
    double dot = 0.0;
    for (std::size_t i = 0; i < n; ++i) dot += h.v[i] * y[i];
    const double scale = h.tau * dot;
    for (std::size_t i = 0; i < n; ++i) y[i] -= scale * h.v[i];
}

} // namespace svd::detail
