#pragma once

#include <flann/flann.h>

namespace mrf {

using DistanceType = flann::L2_Simple<float>;

static constexpr int kdIndexParams = 1;




void getNNdepthPrior(Eigen::VectorXd& depth_est, const Eigen::Matrix3Xd& pts_3d,
                     const Eigen::VectorXi& has_projection, const Eigen::Matrix2Xd& img_pts_raw,
                     const int width, const int height) {




}
}
