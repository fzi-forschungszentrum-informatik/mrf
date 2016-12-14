#include <limits>
#include <ostream>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <generic_logger/generic_logger.hpp>

#include "cloud_preprocessing.hpp"
#include "functor_distance.hpp"
#include "functor_smoothness.hpp"
#include "image_preprocessing.hpp"

namespace mrf {

template <typename T>
bool Solver::solve(Data<T>& data) {

    /**
     * Image preprocessing
     */
    const cv::Mat img{gradientSobel(data.image)};

    /**
     * Cloud transformation and preprocessing
     * \todo Check whether transform is correct or needs to be inverted
     */
    using PointT = pcl::PointXYZINormal;
    const pcl::PointCloud<PointT>::Ptr cl{estimateNormals<T, PointT>(
        transform(data.cloud, data.transform), params_.radius_normal_estimation)};

    /**
     * Compute point projection in camera image
     */
    Eigen::Matrix2Xd img_pts_raw;
    const Eigen::Matrix3Xd pts_3d{cl->getMatrixXfMap().topRows<3>().cast<double>()};
    std::vector<bool> in_img{camera_->getImagePoints(pts_3d, img_pts_raw)};
    int width, height;
    camera_->getImageSize(width, height);
    Eigen::MatrixXi has_projection{-1 * Eigen::MatrixXi::Ones(height, width)};
    for (size_t c = 0; c < in_img.size(); c++) {
        const int row = img_pts_raw(0, c);
        const int col = img_pts_raw(1, c);
        if (in_img[c] && (row > 0) && (row < height) && (col > 0) && (col < width)) {
            has_projection(row, col) = c;
        }
    }

    /**
     * Create optimization problem
     */
    ceres::Problem problem(params_.problem);
    int count_wrong_neighbours{0};
    Eigen::VectorXd depth_est{Eigen::VectorXd::Zero(height * width)};
    std::vector<FunctorDistance::Ptr> functors_distance;
    functors_distance.reserve(height * width);
    Eigen::Quaterniond rotation{data.transform.rotation()};
    Eigen::Vector3d translation{data.transform.translation()};
    for (size_t c = 0; c < in_img.size(); c++) {

        /**
         * Add distance cost if a point can be projected into the image
         */
        functors_distance.emplace_back(
            FunctorDistance::create(Eigen::Vector3d::Zero(), params_.kd));
        problem.AddResidualBlock(functors_distance.back()->toCeres(), new ceres::HuberLoss(1),
                                 &depth_est(c), rotation.coeffs().data(), translation.data());

        /**
         *  Smoothness costs
         */
        for (size_t j = 1; j < 3; j++) {
            const int pnext_lr = c + pow(-1, j);
            const int pnext_tb = c + pow(-1, j) * width;
            const float eij_lr{neighbourDiff(c, pnext_lr, NeighbourCase::left_right)};
            const float eij_tb{neighbourDiff(c, pnext_tb, NeighbourCase::top_bottom)};
            if (eij_lr != -1) { // eij_lr != 0 &&
                problem.AddResidualBlock(FunctorSmoothness::create(eij_lr * params_.ks), nullptr,
                                         &depth_est(c), &depth_est(pnext_lr));
            } else {
                count_wrong_neighbours++;
            }

            if (eij_tb != -1) { // eij_tb != 0 &&
                problem.AddResidualBlock(FunctorSmoothness::create(eij_tb * params_.ks), nullptr,
                                         &depth_est(c), &depth_est(pnext_tb));
            } else {
                count_wrong_neighbours++;
            }
        }
    }
    DEBUG_STREAM("Wrong Neighbours: " << count_wrong_neighbours);

    /**
     * Check parameters
     */
    std::string err_str;
    if (params_.solver.IsValid(&err_str)) {
        INFO_STREAM("All Residuals set up correctly");
    } else {
        ERROR_STREAM(err_str);
    }

    /**
     * Solve problem
     */
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solve(params_.solver, &problem, &summary);
    INFO_STREAM(summary.FullReport());
    return summary.IsSolutionUsable();
}

double Solver::neighbourDiff(const int p, const int pnext, const NeighbourCase& nc, const int width,
                             const int dim) {
    if (((abs((p % width) - (pnext % width)) > 1) || pnext < 0) &&
        nc != NeighbourCase::top_bottom) {
        /*
         * Criteria for left right border pass
         */
        return -1;
    }
    if (((floor(p / width) == 0) && (pnext < 0)) &&
        (nc == NeighbourCase::top_bottom || nc == NeighbourCase::top_left_right)) {
        /*
         * Criteria for top pass
         */
        return -1;
    }
    if ((pnext >= static_cast<int>(dim)) &&
        (nc == NeighbourCase::top_bottom || nc == NeighbourCase::bottom_left_right)) {
        /*
         * Criteria for bottom pass
         */
        return -1;
    }
    return diff(p, pnext);
}

double Solver::diff(const double depth_i, const double depth_j) {
    const float delta{std::abs(depth_i - depth_j)};

    /**
     * \warning Hier wird effektiv nur 1 oder 0 ausgegeben. Das soll wahrscheinlich nicht so sein,
     * oder?
     */
    if (delta < params_.discontinuity_threshold || params_.ks == 0) {
        return 1;
    } else {
        return 0;
    }
    return std::sqrt(1 / (params_.ks * std::sqrt(2 * M_1_PI)) *
                     std::exp(-1 / 2 * std::pow(delta / params_.ks, 2)));
}
}
