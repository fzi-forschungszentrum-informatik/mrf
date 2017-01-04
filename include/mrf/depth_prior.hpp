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

flann::Matrix<double> convertEigen2FlannRow(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mEigen);

bool insideTriangle(const Pixel& p, const Eigen::Vector2i& first, const Eigen::Vector2i& second,
                    const Eigen::Vector2i& third);

std::vector<int> getTriangleNeighbours(std::vector<int>& neighbours_in,
                                       const Eigen::Matrix2Xi& coordinates, const Pixel& p);

std::vector<int> getNeighbours(const Eigen::Matrix2Xi& coordinates,
                               const std::unique_ptr<flann::Index<flann::L2_Simple<double>>>& tree,
                               const Pixel& p, const int num_neigh);

double pointIntersection(const Eigen::ParametrizedLine<double, 3>&,
                         const Eigen::Matrix3Xd& neighbours);

void addSeedPoints(const std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>&,
                   const std::map<Pixel, pcl_ceres::Point<double>, PixelLess>&,
                   Eigen::MatrixXd& depth_est, Eigen::MatrixXd& certainty);

void getDepthEst(const std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>&,
                 const std::map<Pixel, pcl_ceres::Point<double>, PixelLess>&,
                 const size_t& rows, const size_t& cols, const Parameters::Initialization type,
                 const int neighborsearch, Eigen::MatrixXd& depth_est, Eigen::MatrixXd& certainty);
}
