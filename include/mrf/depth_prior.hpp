#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>
#include <flann/flann.h>
#include <pcl_ceres/point.hpp>

#include "eigen.hpp"
#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

using DataType = double;
using Point = pcl_ceres::Point<DataType>;
using mapT = std::map<Pixel, Point, PixelLess>;
using treeT = std::unique_ptr<flann::Index<flann::L2_Simple<DataType>>>;
using EigenT = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

flann::Matrix<DataType> convertEigen2FlannRow(const EigenT& mEigen);

bool insideTriangle(const Pixel& p, const Eigen::Vector2i& first, const Eigen::Vector2i& second,
                    const Eigen::Vector2i& third);

std::vector<int> getTriangleNeighbours(std::vector<int>& neighbours_in,
                                       const Eigen::Matrix2Xi& coordinates, const Pixel& p);

std::vector<int> getNeighbours(const Eigen::Matrix2Xi& coordinates, const treeT& tree,
                               const Pixel& p, const int num_neigh);

double pointIntersection(const Eigen::Vector3d& sp, const Eigen::Vector3d& dir,
                         const Eigen::Matrix3Xd& neighbours);

void addSeedPoints(Eigen::MatrixXd& depth_est, Eigen::MatrixXd& certainty, mapT& projection,
                   const std::shared_ptr<CameraModel> cam);

void getDepthEst(Eigen::MatrixXd& depth_est, Eigen::MatrixXd& certainty, mapT& projection,
                 const std::shared_ptr<CameraModel> cam, const Parameters::Initialization type,
                 const int neighborsearch);
}
