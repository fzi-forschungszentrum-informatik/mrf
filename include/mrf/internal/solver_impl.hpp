#include <limits>
#include <map>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <util_ceres/eigen_quaternion_parameterization.h>

#include "cloud_preprocessing.hpp"
#include "eigen.hpp"
#include "functor_distance.hpp"
#include "functor_smoothness.hpp"
#include "image_preprocessing.hpp"
#include "neighbors.hpp"
#include "smoothness_weight.hpp"

namespace mrf {

template <typename T>
bool Solver::solve(Data<T>& data) {

    LOG(INFO) << "Preprocess image";
    const cv::Mat img{gradientSobel(data.image)};

    LOG(INFO) << "Preprocess and transform cloud";
    using PointT = pcl::PointXYZINormal;

    /**
     * \todo Check if transform is correct or needs to be inverted
     */
    const pcl::PointCloud<PointT>::Ptr cl{estimateNormals<T, PointT>(
        transform<T>(data.cloud, data.transform), params_.radius_normal_estimation)};

    LOG(INFO) << "Compute point projection in camera image";
    const Eigen::Matrix3Xd pts_3d{cl->getMatrixXfMap().topRows<3>().cast<double>()};
    LOG(INFO) << "Rows: " << pts_3d.rows() << ", Cols: " << pts_3d.cols();
    for (size_t col = 0; col < pts_3d.cols(); col++) {
        LOG(INFO) << "Point :" << cl->points[col];
        LOG(INFO) << "Col " << col << ": " << pts_3d.col(col).transpose();
    }

    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, pts_3d.cols())};
    std::vector<bool> in_img{camera_->getImagePoints(pts_3d, img_pts_raw)};
    int cols, rows;
    camera_->getImageSize(cols, rows);
    LOG(INFO) << "Image size: " << cols << " x " << rows;
    std::map<Eigen::Vector2d, Eigen::Vector3d, EigenLess> projection;
    for (size_t c = 0; c < in_img.size(); c++) {
        const int col = img_pts_raw(0, c);
        const int row = img_pts_raw(1, c);
        LOG(INFO) << "row: " << row << ", col: " << col;
        if (in_img[c] && (row > 0) && (row < rows) && (col > 0) && (col < cols)) {
            projection.insert(std::make_pair(img_pts_raw.col(c), pts_3d.col(c)));
        }
    }
    for (auto const& el : projection) {
        LOG(INFO) << "Image coordinate: " << el.first.transpose()
                  << ", 3D point coordinate: " << el.second.transpose();
    }

    LOG(INFO) << "Create optimization problem";
    ceres::Problem problem(params_.problem);
    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    std::vector<FunctorDistance::Ptr> functors_distance;
    functors_distance.reserve(projection.size());
    Eigen::Quaterniond rotation{data.transform.rotation()};
    Eigen::Vector3d translation{data.transform.translation()};
    problem.AddParameterBlock(rotation.coeffs().data(), 4,
                              new util_ceres::EigenQuaternionParameterization);
    problem.AddParameterBlock(translation.data(), 3);

    LOG(INFO) << "Add distance costs";
    for (auto const& el : projection) {
        functors_distance.emplace_back(FunctorDistance::create(el.second, params_.kd));
        problem.AddResidualBlock(
            functors_distance.back()->toCeres(), params_.loss_function.get(),
            &depth_est(static_cast<int>(el.first[1]), static_cast<int>(el.first[0])),
            rotation.coeffs().data(), translation.data());
    }

    LOG(INFO) << "Add smoothness costs and depth limits";
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            const Pixel p(row, col);
            for (auto const& n : getNeighbors(p, rows, cols, params_.neighborhood)) {
                problem.AddResidualBlock(
                    FunctorSmoothness::create(smoothnessWeight(p, n,
                                                               data.image.template at<uchar>(p.col, p.row),
                                                               data.image.template at<uchar>(n.col, n.row)) *
                                              params_.ks),
                    nullptr, &depth_est(row, col), &depth_est(n.row, n.col));
            }
        }
    }

    LOG(INFO) << "Add depth limits";
    if (params_.use_custom_depth_limits) {
        LOG(INFO) << "Use custom limits. Min: " << params_.custom_depth_limit_min
                  << ", Max: " << params_.custom_depth_limit_max;
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {
                problem.SetParameterLowerBound(&depth_est(row, col), 0,
                                               params_.custom_depth_limit_min);
                problem.SetParameterUpperBound(&depth_est(row, col), 0,
                                               params_.custom_depth_limit_max);
            }
        }
    } else {
        const double depth_max{pts_3d.colwise().norm().maxCoeff()};
        const double depth_min{pts_3d.colwise().norm().minCoeff()};
        LOG(INFO) << "Use adaptive limits. Min: " << depth_min << ", Max: " << depth_max;
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {
                problem.SetParameterLowerBound(&depth_est(row, col), 0, depth_min);
                problem.SetParameterUpperBound(&depth_est(row, col), 0, depth_max);
            }
        }
    }

    LOG(INFO) << "Set parameterization";
    problem.SetParameterBlockConstant(rotation.coeffs().data());
    problem.SetParameterBlockConstant(translation.data());

    LOG(INFO) << "Check parameters";
    std::string err_str;
    if (params_.solver.IsValid(&err_str)) {
        LOG(INFO) << "Residuals set up correctly";
    } else {
        LOG(ERROR) << err_str;
    }

    LOG(INFO) << "Solve problem";
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solve(params_.solver, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    data.cloud->clear();
    data.cloud->reserve(rows * cols);
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            LOG(INFO) << "Estimated depth for (" << col << "," << row
                      << "): " << depth_est(row, col);
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(col, row), support, direction);
            T p;
            p.getVector3fMap() =
                (data.transform.inverse() * (support + direction * depth_est(row, col)))
                    .template cast<float>();
            data.cloud->push_back(p);
        }
    }

    return summary.IsSolutionUsable();
}
}
