#include "normal_prior.hpp"

namespace mrf {

template <typename T>
void getNormalEst(Cloud<T>& cl, mapT& projection, const std::shared_ptr<CameraModel> cam) {

    int rows, cols;
    cam->getImageSize(cols, rows);


    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            Eigen::Vector3d support, direction;
            cam->getViewingRay(Eigen::Vector2d(row, col), support, direction);
            cl.at(col, row).normal = -direction;
        }
    }

    for (const auto& el : projection) {
    	cl.at(el.first.col,el.first.row).normal =  el.second.normal;
    }
}
}
