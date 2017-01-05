#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>
#include <pcl_ceres/point.hpp>
#include <pcl_ceres/point_cloud.hpp>

#include "eigen.hpp"
#include "pixel.hpp"

namespace mrf {

template <typename T>
void getNormalEst(pcl_ceres::PointCloud<T>& cl,
                  const std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>& rays,
                  const std::map<Pixel, T, PixelLess>& projection) {

    for (auto const& el : rays) {
        cl.at(el.first.col, el.first.row).normal = -el.second.direction().normalized();
    }

    for (const auto& el : projection) {
        cl.at(el.first.col, el.first.row).normal = el.second.normal;
    }
}
}
