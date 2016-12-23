#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>

#include "cloud.hpp"
#include "eigen.hpp"
#include "pixel.hpp"

namespace mrf {

template <typename T>
void getNormalEst(Cloud<T>& cl, const std::map<Pixel, Point<double>, PixelLess>& projection,
                  const std::shared_ptr<CameraModel>& cam) {

    int rows, cols;
    cam->getImageSize(cols, rows);

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            Eigen::Vector3d support, direction;
            cam->getViewingRay(Eigen::Vector2d(col, row), support, direction);
            cl.at(col, row).normal = -direction.normalized();
        }
    }

    for (const auto& el : projection) {
        cl.at(el.first.col, el.first.row).normal = el.second.normal;
    }
}
}
