#pragma once

#include <ceres/jet.h>

namespace mrf {

template <typename T>
inline T sigmoid(const T& x, const T& theta, const T& m) {
    const T d{ceres::abs(((x - theta) / m - static_cast<T>(5)))};
    return static_cast<T>(0.5) * (static_cast<T>(1) - (d / ceres::sqrt(static_cast<T>(1) + d * d)));
}

template <typename T>
inline T sigmoid(const T& ref, const T& theta1, const T& theta2, const T& m) {
    return sigmoid(ref, theta1, m) * sigmoid(ref, theta2, m);
}
}
