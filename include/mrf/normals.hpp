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
}
