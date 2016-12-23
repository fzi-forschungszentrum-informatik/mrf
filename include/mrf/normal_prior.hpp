#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>
#include <pcl/point_types.h>

#include "eigen.hpp"
#include "parameters.hpp"
#include "pixel.hpp"
#include "cloud.hpp"

namespace mrf {

using PointT = pcl::PointXYZINormal;
using DataType = double;
using mapT = std::map<Pixel, PointT, PixelLess>;

template<typename T>
void getNormalEst(Cloud<T>& cl, mapT& projection,
                  const std::shared_ptr<CameraModel> cam);
}
