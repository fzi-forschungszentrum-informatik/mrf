#pragma once

#include <map>
#include <Eigen/Eigen>

#include "parameters.hpp"
#include "pixel.hpp"
#include "point.hpp"

namespace mrf {

void estimatePrior(const std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>&,
                   const std::map<Pixel, Point<double>, PixelLess>&,
                   const size_t& rows,
                   const size_t& cols,
                   const Parameters&,
                   Eigen::MatrixXd& depth_est,
                   Eigen::MatrixXd& certainty);
}
