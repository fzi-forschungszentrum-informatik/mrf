#pragma once

#include <Eigen/Geometry>
#include "eigen.hpp"

namespace mrf {

template <typename T>
struct Point {

    using ScalarT = T;

    inline Point()
            : position{Eigen::Vector3<T>::Zero()}, normal{Eigen::Vector3<T>::Zero()},
              intensity{0} {};
    inline Point(const Eigen::Vector3<T>& position_,
                 const Eigen::Vector3<T>& normal_,
                 const T& intensity_ = 0)
            : position{position_}, normal{normal_}, intensity{intensity_} {};

    Eigen::Vector3<T> position;
    Eigen::Vector3<T> normal;
    T intensity;
};

void estimatePrior(const std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>&,
                   const std::map<Pixel, Point<double>, PixelLess>&,
                   const size_t& rows,
                   const size_t& cols,
                   const Parameters&,
                   Eigen::MatrixXd& depth_est,
                   Eigen::MatrixXd& certainty);
}
