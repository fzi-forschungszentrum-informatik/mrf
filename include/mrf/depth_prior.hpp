#pragma once

#include <flann/flann.h>

namespace mrf {

using DistanceType = flann::L2_Simple<float>;
static constexpr int kdIndexParams = 1;

void getNNdepthPrior(Eigen::VectorXd& depth_est, const Eigen::Matrix3Xd& pts_3d,
                     const Eigen::VectorXi& has_projection, const Eigen::Matrix2Xd& img_pts_raw,
                     const int width, const int height) {

    /**
     * Build Flann kd Tree
     */
    const int number_points{(has_projection.array() != -1).count()};
    flann::Matrix<DistanceType::ElementType> flann_dataset(2, number_points);
    int i{0};
    for (size_t c = 0; c < has_projection.size(); c++) {
        if (has_projection(c) != -1) {
            flann_dataset[i] = img_pts_raw(c);
            i++;
        }
    }
    std::unique_ptr<flann::Index<DistanceType>> kd_index_ptr_{std::make_unique<
        flann::Index<DistanceType>>(
        flann_dataset,
        flann::KDTreeIndexParams(
            kdIndexParams))}; // KDTreeSingleIndexParams(8)flann::AutotunedIndexParams(0.9,0.01,0,0.1)
    kd_index_ptr_->buildIndex(flann_dataset);

    /**
     * Calc deps
     */

    for(size_t c=0; c<width*height;c++){


    }


}
}
