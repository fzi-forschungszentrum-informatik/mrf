#include <limits>
#include <map>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <util_ceres/eigen_quaternion_parameterization.h>
#include <opencv2/highgui/highgui.hpp>

#include "cloud_preprocessing.hpp"
#include "depth_prior.hpp"
#include "functor_distance.hpp"
#include "functor_smoothness.hpp"
#include "image_preprocessing.hpp"
#include "neighbors.hpp"
#include "smoothness_weight.hpp"

namespace mrf {

template <typename T>
bool Solver::solve(Data<T>& data, const bool pin_transform) {

    LOG(INFO) << "Preprocess image";
    const cv::Mat img{gradientSobel(data.image)};

    LOG(INFO) << "Preprocess and transform cloud";
    using PointT = pcl::PointXYZINormal;
    const pcl::PointCloud<PointT>::Ptr cl{estimateNormals<T, PointT>(
        transform<T>(data.cloud, data.transform), params_.radius_normal_estimation)};

    LOG(INFO) << "Compute point projection in camera image";
    const Eigen::Matrix3Xd pts_3d{cl->getMatrixXfMap().topRows<3>().cast<double>()};

    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, pts_3d.cols())};
    std::vector<bool> in_img{camera_->getImagePoints(pts_3d, img_pts_raw)};
    int cols, rows;
    camera_->getImageSize(cols, rows);
    LOG(INFO) << "Image size: " << cols << " x " << rows;
    std::map<Pixel, Eigen::Vector3d, PixelLess> projection;
    for (size_t c = 0; c < in_img.size(); c++) {
        Pixel p(img_pts_raw(0, c), img_pts_raw(1, c));
        if (in_img[c] && (p.row > 0) && (p.row < rows) && (p.col > 0) && (p.col < cols)) {
            p.val = img.at<float>(p.row, p.col);
            projection.insert(std::make_pair(p, pts_3d.col(c)));
        }
    }

    LOG(INFO) << "Create optimization problem";
    ceres::Problem problem(params_.problem);
    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Ones(rows, cols)};
    getDepthEst(depth_est, certainty,projection, camera_, params_.initialization,params_.neighborsearch);
    LOG(INFO) << "Depth est loaded";
//    for (size_t row = 0; row < rows; row++) {
//        for (size_t col = 0; col < cols; col++) {
//            LOG(INFO) << "Initial depth for: (" << col << "," << row
//                      << "): " << depth_est(row, col);
//        }
//    }
    std::vector<FunctorDistance::Ptr> functors_distance;
    functors_distance.reserve(projection.size());
    Eigen::Quaterniond rotation{data.transform.rotation()};
    Eigen::Vector3d translation{data.transform.translation()};
    problem.AddParameterBlock(rotation.coeffs().data(), FunctorDistance::DimRotation,
                              new util_ceres::EigenQuaternionParameterization);
    problem.AddParameterBlock(translation.data(), FunctorDistance::DimTranslation);

    LOG(INFO) << "Add distance costs";
    for (auto const& el : projection) {
        Eigen::Vector3d support, direction;
        camera_->getViewingRay(Eigen::Vector2d(el.first.x, el.first.y), support, direction);
        functors_distance.emplace_back(
            FunctorDistance::create(el.second, params_.kd, support, direction));
        problem.AddResidualBlock(functors_distance.back()->toCeres(), params_.loss_function.get(),
                                 &depth_est(el.first.row, el.first.col), rotation.coeffs().data(),
                                 translation.data());
    }

    LOG(INFO) << "Add smoothness costs";
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            const Pixel p(col, row, img.at<float>(row, col));
            const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
            const double certainty_sum = neighbors.size();
            for (auto const& n : neighbors) {
                problem.AddResidualBlock(
                    FunctorSmoothness::create(smoothnessWeight(p, n, params_) * params_.ks* certainty(row,col)),
                    new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / certainty_sum,
                                          ceres::TAKE_OWNERSHIP),
                    &depth_est(row, col), &depth_est(n.row, n.col));
            }
        }
    }

    if (params_.limits != Parameters::Limits::none) {
        double ub, lb;
        if (params_.limits == Parameters::Limits::custom) {
            ub = params_.custom_depth_limit_max;
            lb = params_.custom_depth_limit_min;
            LOG(INFO) << "Use custom limits. Min: " << lb << ", Max: " << ub;
        } else if (params_.limits == Parameters::Limits::adaptive) {
            ub = pts_3d.colwise().norm().maxCoeff();
            lb = pts_3d.colwise().norm().minCoeff();
            LOG(INFO) << "Use adaptive limits. Min: " << lb << ", Max: " << ub;
        }
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                problem.SetParameterLowerBound(&depth_est(row, col), 0, lb);
                problem.SetParameterUpperBound(&depth_est(row, col), 0, ub);
            }
        }
    }

    if (pin_transform) {
        LOG(INFO) << "Set parameterization";
        problem.SetParameterBlockConstant(rotation.coeffs().data());
        problem.SetParameterBlockConstant(translation.data());
    }

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
//            LOG(INFO) << "Estimated depth for (" << col << "," << row
//                      << "): " << depth_est(row, col);
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(col, row), support, direction);
            T p;
            p.getVector3fMap() =
                (data.transform.inverse() * (support + direction * depth_est(row, col)))
                    .template cast<float>();
            data.cloud->push_back(p);
        }
    }
    data.cloud->width = cols;
    data.cloud->height = rows;

    Eigen::Affine3d tf_new(rotation);
    tf_new.translation() = translation;
    data.transform = tf_new;

    return summary.IsSolutionUsable();
}
}
