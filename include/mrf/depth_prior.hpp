#pragma once

#include <memory>
#include <flann/flann.h>
#include <Eigen/Eigen>

namespace mrf {

using DistanceType = flann::L2_Simple<double>;

static constexpr int kdIndexParams = 1;

struct DepthPriorTriangleParams {
	int search_x_neighbours{15};

};

class DepthPriorTriangle {
public:
    DepthPriorTriangle(const Eigen::Matrix3Xd& pts_3d,const Eigen::Matrix2Xd& img_pts_raw, const Eigen::VectorXi& has_projection,
                       const int width, const int height,
                       const DepthPriorTriangleParams p = DepthPriorTriangleParams());
    void getDepthEst(Eigen::Vector2d& depth_est);

private:
    std::unique_ptr<flann::Index<DistanceType>> kd_index_ptr_;
    void initTree(const Eigen::MatrixXd&);
    void getNeighbours();
    DepthPriorTriangleParams params_;
};

void getNNdepthPrior(Eigen::VectorXd& depth_est, const Eigen::Matrix3Xd& pts_3d,
                     const Eigen::VectorXi& has_projection, const Eigen::Matrix2Xd& img_pts_raw,
                     const int width, const int height) {
}
}
