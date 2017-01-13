#pragma once

#include <ceres/jet.h>

namespace mrf {

template <typename T>
inline T sigmoid(const T& x, const T& theta, const T& m) {
    const T d{ceres::abs((x - theta) * m)};
    return static_cast<T>(1) - (d / ceres::sqrt(static_cast<T>(1) + d * d));
}
}
