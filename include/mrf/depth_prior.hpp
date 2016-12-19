#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>
#include <flann/flann.h>
#include "eigen.hpp"

namespace mrf {

using DistanceType = flann::L2_Simple<double>;
using mapT = std::map<Eigen::Vector2d, Eigen::Vector3d, EigenLess>;
using treeT = std::unique_ptr<flann::Index<DistanceType>>;
using DataType = double;
using EigenT = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

flann::Matrix<DataType> convertEigen2FlannRow(const EigenT& mEigen) {
    flann::Matrix<DataType> mFlann(new DataType[mEigen.size()], mEigen.rows(), mEigen.cols());

    for (size_t n = 0; n < (unsigned)mEigen.size(); ++n) {
        *(mFlann.ptr() + n) = *(mEigen.data() + n);
    }
    return mFlann;
}

bool insideTriangle(const double& x, const double& y, const Eigen::Vector2d& first,
                    const Eigen::Vector2d& second, const Eigen::Vector2d& third) {

    const Eigen::Vector2d& P{x, y};
    const Eigen::Vector2d& AC{first - third};
    const Eigen::Vector2d& BA{second - first};
    const Eigen::Vector2d& CB{third - second};
    const Eigen::Vector2d& PA{P - first};
    const Eigen::Vector2d& PB{P - second};
    const Eigen::Vector2d& PC{P - third};

    const double u{(BA[0] * PA[1] - (BA[1] * PA[0]))};
    const double v{(BA[0] * (-1 * AC[1])) - (BA[1] * (-1 * AC[0]))};
    if (u * v >= 0) {
        const double u2{(AC[0] * PC[1] - (AC[1] * PC[0]))};
        const double v2{(AC[0] * (-1 * CB[1])) - (AC[1] * (-1 * CB[0]))};
        if (u2 * v2 >= 0) {
            const double u3{(CB[0] * PB[1] - (CB[1] * PB[0]))};
            const double v3{(CB[0] * (-1 * BA[1])) - (CB[1] * (-1 * BA[0]))};
            if (u3 * v3 >= 0) {
                return true;
            }
        }
    }
    return false;
}

std::vector<int> getNeighbours(Eigen::Matrix2Xd& coordinates, const treeT& tree, const int u,
                               const int v, const int num_neigh) {
    std::vector<int> neighbours{0, 1, 2};
    DistanceType::ElementType queryData[] = {static_cast<DistanceType::ElementType>(u),
                                             static_cast<DistanceType::ElementType>(v)};

    const flann::Matrix<DistanceType::ElementType> query(queryData, 1, 2);
    std::vector<std::vector<int>> indices_vec;
    std::vector<std::vector<double>> dist_vec;
    tree->knnSearch(query, indices_vec, dist_vec, num_neigh, flann::SearchParams(8));

    int i = 0;
    while (i < (num_neigh - 2)) {
        const Eigen::Vector2d& first_coorinate{coordinates.col(indices_vec[0][i])};
        neighbours[0] = indices_vec[0][i];
        const Eigen::Vector2d& second_coorinate{coordinates.col(indices_vec[0][i + 1])};
        neighbours[1] = indices_vec[0][i + 1];
        int n = 2;
        while (n < num_neigh) {
            const Eigen::Vector2d& third_coorinate{coordinates.col(indices_vec[0][i + 1])};
            neighbours[2] = indices_vec[0][i + n];
            if (insideTriangle(u, v, first_coorinate, second_coorinate, third_coorinate)) {
                return neighbours;
            } else {
                n++;
            }
        }
        i++;
    }
    return std::vector<int>{-1, -1, -1};
}

double pointIntersection(const Eigen::Vector3d& sp, const Eigen::Vector3d& dir,
                         const Eigen::Matrix3Xd& neighbours) {
    const Eigen::Vector3d& q0_world = neighbours.col(0);
    const Eigen::Vector3d& q1_world = neighbours.col(1);
    const Eigen::Vector3d& q2_world = neighbours.col(2);

    Eigen::Vector3d direction_1_norm = (q0_world - q1_world); //.normalized();
    Eigen::Vector3d direction_2_norm = (q2_world - q1_world); //.normalized();

    Eigen::Vector3d normal = (direction_1_norm.cross(direction_2_norm)).normalized();

    const Eigen::ParametrizedLine<double, 3> pline(sp, dir);
    Eigen::Hyperplane<double, 3> plane(normal, q1_world);
    Eigen::Vector3d p_int{pline.intersectionPoint(plane)};
    return p_int.norm();
}

void getNNDepthEst(Eigen::MatrixXd& depth_est, mapT projection,
                   const std::shared_ptr<CameraModel> camera, const int width, const int height) {

    std::unique_ptr<flann::Index<DistanceType>> kd_index_ptr_;

    Eigen::Matrix2Xd coordinates(2, projection.size());
    int i{0};
    for (mapT::iterator it = projection.begin(); it != projection.end(); ++it) {
        coordinates.col(i) = (it->first);
        i++;
    }

    flann::Matrix<DistanceType::ElementType> flann_dataset{
        convertEigen2FlannRow(coordinates)}; //>todo:: Check whether colum or row major
    kd_index_ptr_ =
        std::make_unique<flann::Index<DistanceType>>(flann_dataset, flann::KDTreeIndexParams(8));
    kd_index_ptr_->buildIndex(flann_dataset);

    for (size_t v = 0; v < height; v++) {
        for (size_t u = 0; u < width; u++) {
            std::vector<int> neighbours{getNeighbours(coordinates, kd_index_ptr_, u, v, 15)};
            if (neighbours[0] == -1) {
                depth_est(v * width + u) = -1;
            } else {
                Eigen::Vector3d supportPoint(3, 1);
                Eigen::Vector3d direction(3, 1);
                Eigen::Vector2d image_coordinate{u, v};
                Eigen::Matrix3Xd neighbours_points(3, 3);
                for (size_t i = 0; i < 3; i++) {
                    Eigen::Vector2d c{coordinates.col(neighbours[i])};

                    neighbours_points.col(i) = projection.at(c);
                }
                camera->getViewingRay(image_coordinate, supportPoint, direction);
                depth_est(v * width + u) =
                    pointIntersection(supportPoint, direction, neighbours_points);
            }
        }
    }
}
}
