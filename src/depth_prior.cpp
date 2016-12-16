#include "depth_prior.hpp"

namespace mrf {


using DataType = double;
using EigenT = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline flann::Matrix<DataType> convertEigen2FlannRow(const EigenT& mEigen) {
    flann::Matrix<DataType> mFlann(new DataType[mEigen.size()], mEigen.rows(), mEigen.cols());

    for (size_t n = 0; n < (unsigned)mEigen.size(); ++n) {
        *(mFlann.ptr() + n) = *(mEigen.data() + n);
    }
    return mFlann;
}


DepthPriorTriangle::DepthPriorTriangle(
    const Eigen::Matrix3Xd& pts_3d, const Eigen::Matrix2Xd& img_pts_raw,
    const Eigen::VectorXi& has_projection, const int width, const int height,
    const DepthPriorTriangleParams p)
        : params_{p} {
    const int n{(has_projection.array() != -1).count()};
    Eigen::MatrixXd coordinates{Eigen::MatrixXd::Zero(2, n)};
    int i{0};
    for (size_t c = 0; c < has_projection.size(); c++) {
        if (has_projection(c) != -1) {
            coordinates(0, i) = img_pts_raw(0, has_projection(c));
            coordinates(1, i) = img_pts_raw(1, has_projection(c));
            i++;
        }
    }
    initTree(coordinates);

    Eigen::VectorXd depths_est{-1 * Eigen::VectorXd::Ones(width * height)};
    for (size_t c = 0; c < has_projection.size(); c++) {
        if (has_projection(c) == -1) {
            const int row{c % width};
            const int col{floor(c / width)};
            DistanceType::ElementType queryData[] = {
                static_cast<DistanceType::ElementType>(c % width),
                static_cast<DistanceType::ElementType>(floor(c / width))};
            const flann::Matrix<DistanceType::ElementType> query(queryData, 1, 2);
            std::vector<std::vector<int>> indices_vec;
            std::vector<std::vector<double>> dist_vec;
            kd_index_ptr_->knnSearch(query, indices_vec, dist_vec, params_.search_x_neighbours,
                                     flann::SearchParams(32));
            Eigen::Matrix2Xf neigh_coordinates(2, params_.search_x_neighbours);

        } else {
            depths_est(c) = pts_3d.col(has_projection(c)).norm();
        }
    }
}

void DepthPriorTriangle::initTree(const Eigen::MatrixXd& coordinates) {
    flann::Matrix<DistanceType::ElementType> flann_dataset{
        convertEigen2FlannRow(coordinates)}; //>todo:: Check whether colum or row major

    kd_index_ptr_ =
        std::make_unique<flann::Index<DistanceType>>(flann_dataset, flann::KDTreeIndexParams(8));
    kd_index_ptr_->buildIndex(flann_dataset);
}
}
