#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>
#include <flann/flann.h>

namespace mrf {

using DistanceType = flann::L2_Simple<double>;
using mapT = std::map<Eigen::Vector2d, Eigen::Vector3d, EigenLess>;

static constexpr int kdIndexParams = 1;

struct DepthPriorTriangleParams {
    int search_x_neighbours{15};
};

class DepthPriorTriangle {
public:
    DepthPriorTriangle(const Eigen::Matrix3Xd& pts_3d, const Eigen::Matrix2Xd& img_pts_raw,
                       const Eigen::VectorXi& has_projection, const int width, const int height,
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

void getNNDepthEst(const mapT projection, const std::shared_ptr<CameraModel> camera, const int width, const int height) {

    std::unique_ptr<flann::Index<DistanceType>> kd_index_ptr_;
    Eigen::Matrix2Xd coordinates(projection.size());
    int i{0};
    for (mapT::iterator it = projection.begin(); it != projection.end(); ++it) {
        coordinates(i)(it->first);
        i++;
    }
    flann::Matrix<DistanceType::ElementType> flann_dataset{
        convertEigen2FlannRow(coordinates)}; //>todo:: Check whether colum or row major
    kd_index_ptr_ =
        std::make_unique<flann::Index<DistanceType>>(flann_dataset, flann::KDTreeIndexParams(8));
    kd_index_ptr_->buildIndex(flann_dataset);

    /*
     * Iterate through all poits
     *
     */



}
}
