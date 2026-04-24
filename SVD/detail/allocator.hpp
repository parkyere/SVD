#pragma once
//
// 내부: 64-byte 정렬 메모리 할당 + 크기 안전 가드.
// 사용자가 include할 일 없음.
//

#include "../svd.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <limits>
#include <memory>
#include <new>

namespace svd::detail {

constexpr std::size_t MAX_MATRIX_BYTES = SVD_MAX_MATRIX_BYTES;
constexpr std::size_t MAX_MATRIX_DOUBLES = MAX_MATRIX_BYTES / sizeof(double);

// 검증된 곱셈: size_t overflow 시 length_error.
inline std::size_t safe_mul(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("matrix dimension multiplication overflow");
    return a * b;
}

inline void check_element_count(std::size_t count) {
    if (count > MAX_MATRIX_DOUBLES) {
        throw std::length_error(
            "matrix allocation exceeds safety limit ("
            + std::to_string(MAX_MATRIX_BYTES >> 30) + " GiB; requested "
            + std::to_string(count * sizeof(double)) + " bytes)");
    }
}

// AlignedFree 정의는 svd.hpp에 (EBO 요구로 complete type 필요).
// 여기서는 이미 보이는 detail::AlignedDoublePtr만 사용.

inline AlignedDoublePtr allocate_aligned_uninitialized(std::size_t count) {
    if (count == 0) return nullptr;
    check_element_count(count);                                      // 1차: 안전 상한
    const std::size_t bytes = safe_mul(count, sizeof(double));       // 2차: overflow
    const std::size_t alloc_bytes = (bytes + 63) & ~std::size_t{ 63 };
    if (alloc_bytes < bytes)                                         // 3차: 반올림 wrap
        throw std::length_error("matrix allocation byte rounding overflow");
    void* ptr = ::operator new(alloc_bytes, std::align_val_t{ 64 });
    return AlignedDoublePtr(static_cast<double*>(ptr));
}

// =======================================================================
// MatrixAccess — Matrix의 private raw 포인터·stride 접근의 유일한 게이트.
// 내부 SVD 알고리즘만 사용. 사용자 코드는 이 헤더를 include하지 않으므로
// 이 클래스의 정의를 볼 수 없다.
// =======================================================================
class MatrixAccess {
public:
    static double* data(Matrix& m) noexcept { return m.data_.get(); }
    static const double* data(const Matrix& m) noexcept { return m.data_.get(); }
    static std::size_t stride(const Matrix& m) noexcept { return m.stride_; }
};

} // namespace svd::detail
