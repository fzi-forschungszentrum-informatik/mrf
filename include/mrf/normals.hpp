#pragma once

#include <map>
#include <vector>
#include <util_ceres/eigen.h>

#include "math.hpp"
#include "pixel.hpp"

namespace mrf {

template <typename T>
Eigen::Vector3<T> estimateNormal(
    const Pixel& ref,
    const std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>& rays,
    const std::map<Pixel, T, PixelLess>& depths,
    const std::vector<std::pair<Pixel, Pixel>>& mapping,
    const T& m) {

    using namespace Eigen;

    const Vector3<T> p_ref{rays.at(ref).cast<T>().pointAt(depths.at(ref))};
    Vector3<T> n{Vector3<T>::Zero()};
    for (auto const& el : mapping) {
        const Vector3<T> p_neighbor1{rays.at(el.first).cast<T>().pointAt(depths.at(el.first))};
        const Vector3<T> p_neighbor2{rays.at(el.second).cast<T>().pointAt(depths.at(el.second))};
        n = n +
            sigmoid(depths.at(ref), depths.at(el.first), m) *
                sigmoid(depths.at(ref), depths.at(el.first), m) *
                (p_neighbor1 - p_ref).cross(p_neighbor2 - p_ref).normalized();
    }
    return n.normalized();
}

template <typename T>
Eigen::Vector3<T> estimateNormal4(const T& d_ref,
                                 const Eigen::ParametrizedLine<T, 3>& ray_ref,
                                 const T& d_top,
                                 const Eigen::ParametrizedLine<T, 3>& ray_top,
                                 const T& d_right,
                                 const Eigen::ParametrizedLine<T, 3>& ray_right,
                                 const T& d_bottom,
                                 const Eigen::ParametrizedLine<T, 3>& ray_bottom,
                                 const T& d_left,
                                 const Eigen::ParametrizedLine<T, 3>& ray_left,
                                 const T& m) {
    using namespace Eigen;

    const Vector3<T> p_ref{ray_ref.pointAt(d_ref)};
    const Vector3<T> diff_top{ray_top.pointAt(d_top) - p_ref};
    const Vector3<T> diff_right{ray_right.pointAt(d_right) - p_ref};
    const Vector3<T> diff_bottom{ray_bottom.pointAt(d_bottom) - p_ref};
    const Vector3<T> diff_left{ray_left.pointAt(d_left) - p_ref};

    const Vector3<T> n_top_left{diff_top.cross(diff_left).normalized()};
    const Vector3<T> n_right_top{diff_right.cross(diff_top).normalized()};
    const Vector3<T> n_bottom_right{diff_bottom.cross(diff_right).normalized()};
    const Vector3<T> n_left_bottom{diff_left.cross(diff_bottom).normalized()};

    const Vector3<T> n{sigmoid(d_ref, d_top, d_left, m) * n_top_left +
                       sigmoid(d_ref, d_right, d_top, m) * n_right_top +
                       sigmoid(d_ref, d_bottom, d_right, m) * n_bottom_right +
                       sigmoid(d_ref, d_left, d_bottom, m) * n_left_bottom};
    return n.normalized();
}
}
