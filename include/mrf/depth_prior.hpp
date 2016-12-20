#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>
#include <flann/flann.h>
#include "eigen.hpp"
#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

using DataType = double;
using DistanceType = flann::L2_Simple<DataType>;
using mapT = std::map<Pixel, Eigen::Vector3d, PixelLess>;
using treeT = std::unique_ptr<flann::Index<DistanceType>>;
using EigenT = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

flann::Matrix<DataType> convertEigen2FlannRow(const EigenT& mEigen) {
    flann::Matrix<DataType> mFlann(new DataType[mEigen.size()], mEigen.rows(), mEigen.cols());

    for (size_t n = 0; n < (unsigned)mEigen.size(); ++n) {
        *(mFlann.ptr() + n) = *(mEigen.data() + n);
    }
    return mFlann;
}

bool insideTriangle(const Pixel& p, const Eigen::Vector2i& first, const Eigen::Vector2i& second,
                    const Eigen::Vector2i& third) {
    const Eigen::Vector2i& P{p.col, p.row};
    const Eigen::Vector2i& AC{first - third};
    const Eigen::Vector2i& BA{second - first};
    const Eigen::Vector2i& CB{third - second};
    const Eigen::Vector2i& PA{P - first};
    const Eigen::Vector2i& PB{P - second};
    const Eigen::Vector2i& PC{P - third};

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

std::vector<int> getTriangleNeighbours(std::vector<int>& neighbours_in,
                                       const Eigen::Matrix2Xi& coordinates, const Pixel& p) {

    std::vector<int> neighbours{-1, -1, -1};
    int i{0};
    while (i < 3) { //(neighbours_in.size() - 2)
        const Eigen::Vector2i& first_coorinate{coordinates(0, neighbours_in[i]),
                                               coordinates(1, neighbours_in[i])};
        neighbours[0] = neighbours_in[i];
        int j{1 + i};
        while (j < (neighbours_in.size() - 1)) {
            const Eigen::Vector2i& second_coorinate{coordinates(0, neighbours_in[j]),
                                                    coordinates(1, neighbours_in[j])};
            neighbours[1] = neighbours_in[j];
            int n{j + 1};
            while (n < (neighbours_in.size())) {
                const Eigen::Vector2i& third_coorinate{coordinates(0, neighbours_in[n]),
                                                       coordinates(1, neighbours_in[n])};
                neighbours[2] = neighbours_in[n];
                if (insideTriangle(p, first_coorinate, second_coorinate, third_coorinate)) {
                    LOG(INFO) << "Found point inside triangle: at iteration i=" << i << ", j= " << j
                              << ", n=" << n << "\n"
                              << "Point: " << p << "Neighbour coordinates: (" << first_coorinate.x()
                              << ", " << first_coorinate.y() << ") , (" << second_coorinate.x()
                              << ", " << second_coorinate.y() << ") , (" << third_coorinate.x()
                              << ", " << third_coorinate.y() << ") ";
                    return neighbours;
                }
                n++;
            }
            j++;
        }
        i++;
    }
    return std::vector<int>{-1, -1, -1};
}

std::vector<int> getNeighbours(const Eigen::Matrix2Xi& coordinates, const treeT& tree,
                               const Pixel& p, const int num_neigh) {
    DistanceType::ElementType queryData[] = {static_cast<DistanceType::ElementType>(p.col),
                                             static_cast<DistanceType::ElementType>(p.row)};

    const flann::Matrix<DistanceType::ElementType> query(queryData, 1, 2);
    std::vector<std::vector<int>> indices_vec;
    std::vector<std::vector<DataType>> dist_vec;
    tree->knnSearch(query, indices_vec, dist_vec, num_neigh, flann::SearchParams(32));

    return indices_vec[0];
}

double pointIntersection(const Eigen::Vector3d& sp, const Eigen::Vector3d& dir,
                         const Eigen::Matrix3Xd& neighbours) {
    const Eigen::Vector3d& q0_world{neighbours.col(0)};
    const Eigen::Vector3d& q1_world{neighbours.col(1)};
    const Eigen::Vector3d& q2_world{neighbours.col(2)};

    Eigen::Vector3d direction_1_norm{q0_world - q1_world}; //.normalized();
    Eigen::Vector3d direction_2_norm{q2_world - q1_world}; //.normalized();

    Eigen::Vector3d normal = (direction_1_norm.cross(direction_2_norm)).normalized();

    const Eigen::ParametrizedLine<double, 3> pline(sp, dir);
    Eigen::Hyperplane<double, 3> plane(normal, q0_world);
    Eigen::Vector3d p_int{pline.intersectionPoint(plane)};
    return (p_int-sp).norm();
}

void getDepthEst(Eigen::MatrixXd& depth_est, mapT projection,
                 const std::shared_ptr<CameraModel> cam, const Parameters::Initialization type) {
    if (type == Parameters::Initialization::none)
        return;
    int rows, cols;
    cam->getImageSize(cols, rows);
    std::unique_ptr<flann::Index<DistanceType>> kd_index_ptr_;

    Eigen::Matrix2Xi coordinates(2, projection.size());
    int i{0};
    for (mapT::iterator it = projection.begin(); it != projection.end(); ++it) {
        coordinates(0, i) = (it->first).col;
        coordinates(1, i) = (it->first).row;
        i++;
    }

    if (type == Parameters::Initialization::mean_depth) {
        Eigen::Matrix3Xd points(2, projection.size());
        int i{0};
        for (mapT::iterator it = projection.begin(); it != projection.end(); ++it) {
            points.col(i) = (it->second);
            i++;
        }
        double mean_depth{points.colwise().norm().mean()};
        depth_est = mean_depth * Eigen::MatrixXd::Ones(rows, cols);
        return;
    }

    flann::Matrix<DistanceType::ElementType> flann_dataset{
        convertEigen2FlannRow(coordinates.transpose().cast<DataType>())};
    kd_index_ptr_ =
        std::make_unique<flann::Index<DistanceType>>(flann_dataset, flann::KDTreeIndexParams(8));
    kd_index_ptr_->buildIndex(flann_dataset);

    if (type == Parameters::Initialization::nearest_neighbor) {
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {

                std::vector<int> neighbours{
                    getNeighbours(coordinates, kd_index_ptr_, Pixel(col, row), 1)};
                Eigen::Vector3d p{projection.at(
                    Pixel(coordinates(0, neighbours[0]), coordinates(1, neighbours[0])))};
                Eigen::Vector3d support, direction;
                Eigen::Vector2d coor{
                    Eigen::Vector2i(coordinates(0, neighbours[0]), coordinates(1, neighbours[0]))
                        .cast<double>()};
                cam->getViewingRay(coor, support, direction);
                const Eigen::Hyperplane<double, 3> plane(direction, p);
                depth_est(row, col) = (Eigen::ParametrizedLine<double, 3>(support, direction)
                                           .intersectionPoint(plane) -
                                       support)
                                          .norm();
            }
        }
        return;
    }

    if (type == Parameters::Initialization::triangles) {

        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {

                std::vector<int> all_neighbours{
                    getNeighbours(coordinates, kd_index_ptr_, Pixel(col, row),
                                  4)}; //> todo:: 15 to variable parameter value
                std::vector<int> triangle_neighbours{
                    getTriangleNeighbours(all_neighbours, coordinates, Pixel(col, row))};
                if (triangle_neighbours[0] == -1) {
                    depth_est(row, col) = -1;
                } else {
                    Eigen::Vector3d supportPoint(3, 1);
                    Eigen::Vector3d direction(3, 1);
                    Eigen::Vector2d image_coordinate{col, row};
                    Eigen::Matrix3Xd neighbours_points(3, 3);
                    for (size_t i = 0; i < 3; i++) {
                        Pixel c{coordinates(0, triangle_neighbours[i]),
                                coordinates(1, triangle_neighbours[i])};
                        neighbours_points.col(i) = projection.at(c);
                    }
                    cam->getViewingRay(image_coordinate, supportPoint, direction);
                    depth_est(row, col) =
                        pointIntersection(supportPoint, direction, neighbours_points);
                }
            }
        }
    }
}
}
