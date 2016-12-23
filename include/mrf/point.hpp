#pragma once

#include <util_ceres/eigen.h>

namespace mrf {

template <typename T>
struct Point {

    inline Point()
            : position{Eigen::Vector3<T>::Zero()}, normal{Eigen::Vector3<T>::Zero()},
              intensity{0} {};
    inline Point(const Eigen::Vector3<T>& position_, const Eigen::Vector3<T>& normal_,
                 const T& intensity_ = 0)
            : position{position_}, normal{normal_}, intensity{intensity_} {};

    Eigen::Vector3<T> position;
    Eigen::Vector3<T> normal;
    T intensity;
};
}
