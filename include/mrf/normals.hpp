#pragma once

#include <map>
#include <vector>
#include <util_ceres/eigen.h>

#include "math.hpp"
#include "pixel.hpp"

namespace mrf {

template <typename T>
Eigen::Vector3<T> estimateNormal1(const T& d_0,
                                  const Eigen::ParametrizedLine<T, 3>& ray_0,
                                  const T& d_1,
                                  const Eigen::ParametrizedLine<T, 3>& ray_1,
                                  const T& d_2,
                                  const Eigen::ParametrizedLine<T, 3>& ray_2,
                                  const T& m) {
    const Eigen::Vector3<T> p_0{ray_0.pointAt(d_0)};
    return (ray_1.pointAt(d_1) - p_0).cross(ray_2.pointAt(d_2) - p_0).normalized();
}

template <typename T>
Eigen::Vector3<T> estimateNormal2(const T& d_0,
                                  const Eigen::ParametrizedLine<T, 3>& ray_0,
                                  const T& d_1,
                                  const Eigen::ParametrizedLine<T, 3>& ray_1,
                                  const T& d_2,
                                  const Eigen::ParametrizedLine<T, 3>& ray_2,
                                  const T& d_3,
                                  const Eigen::ParametrizedLine<T, 3>& ray_3,
                                  const T& m) {
    using namespace Eigen;
    const Vector3<T> p_0{ray_0.pointAt(d_0)};
    const Vector3<T> diff_1{ray_1.pointAt(d_1) - p_0};
    const Vector3<T> diff_2{ray_2.pointAt(d_2) - p_0};
    const Vector3<T> diff_3{ray_3.pointAt(d_3) - p_0};
    const Vector3<T> n{diff_1.cross(diff_2).normalized() + diff_2.cross(diff_3).normalized()};
    return n.normalized();
}

template <typename T>
Eigen::Vector3<T> estimateNormal4(const T& d_0,
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
    using namespace Eigen;
    const Vector3<T> p_0{ray_0.pointAt(d_0)};
    const Vector3<T> diff_1{ray_1.pointAt(d_1) - p_0};
    const Vector3<T> diff_2{ray_2.pointAt(d_2) - p_0};
    const Vector3<T> diff_3{ray_3.pointAt(d_3) - p_0};
    const Vector3<T> diff_4{ray_4.pointAt(d_4) - p_0};
    const Vector3<T> n_12{diff_1.cross(diff_2).normalized()};
    const Vector3<T> n_23{diff_2.cross(diff_3).normalized()};
    const Vector3<T> n_34{diff_3.cross(diff_4).normalized()};
    const Vector3<T> n_41{diff_4.cross(diff_1).normalized()};
    return (n_12 + n_23 + n_34 + n_41).normalized();
}
}
