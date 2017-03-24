#pragma once

#include <map>
#include <vector>
#include <Eigen/Geometry>

#include "math.hpp"
#include "pixel.hpp"


namespace Eigen {
template <typename T>
using Affine3 = Transform<T, 3, Affine>;

template <typename T>
using Vector3 = Matrix<T, 3, 1>;
}

namespace mrf {


template <typename T>
inline Eigen::Vector3<T> estimateNormal1(const Eigen::Vector3<T>& p_0,
                                         const Eigen::Vector3<T>& p_1,
                                         const Eigen::Vector3<T>& p_2) {
    return (p_1 - p_0).cross(p_2 - p_0).normalized();
}

template <typename T>
inline Eigen::Vector3<T> estimateNormal1(const T& d_0,
                                         const Eigen::ParametrizedLine<T, 3>& ray_0,
                                         const T& d_1,
                                         const Eigen::ParametrizedLine<T, 3>& ray_1,
                                         const T& d_2,
                                         const Eigen::ParametrizedLine<T, 3>& ray_2) {
    const Eigen::Vector3<T> p_0{ray_0.pointAt(d_0)};
    return (ray_1.pointAt(d_1) - p_0).cross(ray_2.pointAt(d_2) - p_0).normalized();
}

template <typename T>
inline Eigen::Vector3<T> estimateNormal2(const T& d_0,
                                         const Eigen::ParametrizedLine<T, 3>& ray_0,
                                         const T& d_1,
                                         const Eigen::ParametrizedLine<T, 3>& ray_1,
                                         const T& d_2,
                                         const Eigen::ParametrizedLine<T, 3>& ray_2,
                                         const T& d_3,
                                         const Eigen::ParametrizedLine<T, 3>& ray_3,
                                         const T& m) {
    return (sigmoid(d_0, d_1, d_2, m) * estimateNormal1(d_0, ray_0, d_1, ray_1, d_2, ray_2) +
            sigmoid(d_0, d_2, d_3, m) * estimateNormal1(d_0, ray_0, d_2, ray_2, d_3, ray_3))
        .normalized();
}

template <typename T>
inline Eigen::Vector3<T> estimateNormal4(const T& d_0,
                                         const Eigen::ParametrizedLine<T, 3>& ray_0,
                                         const T& d_1,
                                         const Eigen::ParametrizedLine<T, 3>& ray_1,
                                         const T& d_2,
                                         const Eigen::ParametrizedLine<T, 3>& ray_2,
                                         const T& d_3,
                                         const Eigen::ParametrizedLine<T, 3>& ray_3,
                                         const T& d_4,
                                         const Eigen::ParametrizedLine<T, 3>& ray_4,
                                         const T& m) {
    return (estimateNormal2(d_0, ray_0, d_1, ray_1, d_2, ray_2, d_3, ray_3, m) +
            estimateNormal2(d_0, ray_0, d_3, ray_3, d_4, ray_4, d_1, ray_1, m))
        .normalized();
}
}
