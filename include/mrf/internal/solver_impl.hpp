#include <limits>
#include <ostream>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <generic_logger/generic_logger.hpp>
#include <util_ceres/eigen_quaternion_parametrization.h>

#include "cloud_preprocessing.hpp"
#include "functor_distance.hpp"
#include "functor_smoothness.hpp"
#include "image_preprocessing.hpp"
#include "neighbours.hpp"

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
    const double depth_max_in{pts_3d.colwise().norm().maxCoeff()};
    const double depth_min_in{pts_3d.colwise().norm().minCoeff()};
    std::vector<bool> in_img{camera_->getImagePoints(pts_3d, img_pts_raw)};
    int width, height;
    camera_->getImageSize(width, height);
    Eigen::VectorXi has_projection{-1 * Eigen::VectorXi::Ones(height * width)};
    for (size_t c = 0; c < in_img.size(); c++) {
        const int row = img_pts_raw(0, c);
        const int col = img_pts_raw(1, c);
        if (in_img[c] && (row > 0) && (row < height) && (col > 0) && (col < width)) {
            const int indice{row * width + col};
            has_projection(indice) = c;
        }
    }

    /**
     * Create optimization problem
     */
    ceres::Problem problem(params_.problem);
    Eigen::VectorXd depth_est{Eigen::VectorXd::Zero(height * width)};

    /**
     * Load Depth Initialisation
     */
    std::vector<FunctorDistance::Ptr> functors_distance;
    functors_distance.reserve(height * width);
    Eigen::Quaterniond rotation{data.transform.rotation()};
    Eigen::Vector3d translation{data.transform.translation()};
    for (size_t c = 0; c < width * height; c++) {

        /**
         * Add distance cost if a point can be projected into the image
         */
        Eigen::Vector3d p{Eigen::Vector3d::Zero()};
        int w{0};
        if (has_projection(c) != -1) {
            p = pts_3d.col(has_projection(c));
            w = params_.kd;
        }

        functors_distance.emplace_back(FunctorDistance::create(p, w));
        problem.AddResidualBlock(functors_distance.back()->toCeres(), new ceres::HuberLoss(1),
                                 &depth_est(c), rotation.coeffs().data(), translation.data());

        /**
         *  Smoothness costs
         */
        std::vector<int> neighbours{getNeighbours(c, params_, width, width * height)};
        std::vector<int> weight{smoothnessWeights(c, neighbours, img)};
        for (size_t i = 0; i < neighbours.size(); i++) {
            	if(neighbours(i) != -1){
            		problem.AddResidualBlock(FunctorSmoothness::create(weight(i) * params_.ks), nullptr,
            		                                         &depth_est(c), &depth_est(neighbours(i)));
            	}
        }
        /**
         * Limits
         */
        if (params_.set_depth_limits) {
            problem.SetParameterLowerBound(&depth_est(c), 0, depth_min_in);
            problem.SetParameterUpperBound(&depth_est(c), 0, depth_max_in);
        }
    }

    /**
     * Parametriztation and boundaries
     */
    problem.SetParameterization(rotation.coeffs().data(),
                                new util_ceres::EigenQuaternionParametrization);

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

//double Solver::neighbourDiff(const int p, const int pnext, const NeighbourCase& nc, const int width,
//                             const int dim) {
//    if (((abs((p % width) - (pnext % width)) > 1) || pnext < 0) &&
//        nc != NeighbourCase::top_bottom) {
//        /*
//         * Criteria for left right border pass
//         */
//        return -1;
//    }
//    if (((floor(p / width) == 0) && (pnext < 0)) &&
//        (nc == NeighbourCase::top_bottom || nc == NeighbourCase::top_left_right)) {
//        /*
//         * Criteria for top pass
//         */
//        return -1;
//    }
//    if ((pnext >= static_cast<int>(dim)) &&
//        (nc == NeighbourCase::top_bottom || nc == NeighbourCase::bottom_left_right)) {
//        /*
//         * Criteria for bottom pass
//         */
//        return -1;
//    }
//    return diff(p, pnext);
//}
//
//double Solver::diff(const double depth_i, const double depth_j) {
//    const float delta{std::abs(depth_i - depth_j)};
//
//    /**
//     * \warning Hier wird effektiv nur 1 oder 0 ausgegeben. Das soll wahrscheinlich nicht so sein,
//     * oder?
//     */
//    if (delta < params_.discontinuity_threshold || params_.ks == 0) {
//        return 1;
//    } else {
//        return 0;
//    }
//    return std::sqrt(1 / (params_.ks * std::sqrt(2 * M_1_PI)) *
//                     std::exp(-1 / 2 * std::pow(delta / params_.ks, 2)));
//}

std::vector<double> smoothnessWeights(const int p, const std::vector<int>& neighbs,
                                      const cv::Mat& img) {

    const int width{img.rows};
    std::vector<double> w(neighbs.size(), 0);
    const int p_row{p % width};
    const int p_col{floor(p / width).cast<int>};

    for (size_t c; c < neighbs.size(); c++) {
        if (neighbs(c) == -1) {
            w(c) = 0;
        } else {
            const int row{neighbs(c) % width};
            const int col{floor(neighbs(c) / width).cast<int>};
            const double delta{
                abs(img.at<uchar>(col, row) - img.at<uchar>(p_col, p_row)).cast<double>()};
            w(c) = delta;
        }

        //> Todo:: Other mathematical function
    }
    return w;
}
}
