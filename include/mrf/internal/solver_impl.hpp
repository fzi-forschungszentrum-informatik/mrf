#include <limits>
#include <map>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <util_ceres/constant_length_parameterization.h>
#include <util_ceres/eigen_quaternion_parameterization.h>
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

#include "../cloud.hpp"
#include "../cloud_preprocessing.hpp"
#include "../depth_prior.hpp"
#include "../functor_distance.hpp"
#include "../functor_normal.hpp"
#include "../functor_normal_distance.hpp"
#include "../functor_smoothness_distance.hpp"
#include "../functor_smoothness_normal.hpp"
#include "../image_preprocessing.hpp"
#include "../neighbors.hpp"
#include "../normal_prior.hpp"
#include "../smoothness_weight.hpp"

namespace mrf {

template <typename T>
ResultInfo Solver::solve(const Data<T>& in, Data<PointT>& out, const bool pin_transform) {

    LOG(INFO) << "Preprocess image";
    const cv::Mat img{edge(in.image)};

    LOG(INFO) << "Preprocess and transform cloud";
    using CloudT = pcl::PointCloud<PointT>;
    CloudT::Ptr cl{new CloudT};
    pcl::copyPointCloud<T, PointT>(*(in.cloud), *cl);
    if (params_.estimate_normals) {
        cl = estimateNormals<PointT, PointT>(cl, params_.radius_normal_estimation);
    }
    std::vector<int> indices;
    pcl::removeNaNNormalsFromPointCloud(*cl, *cl, indices);

    using PType = Point<double>;
    const Cloud<PType>::Ptr cloud{Cloud<PType>::create()};
    cloud->points.reserve(cl->size());
    for (size_t c = 0; c < cl->size(); c++) {
        cloud->points.emplace_back(PType(cl->points[c].getVector3fMap().cast<double>(),
                                         cl->points[c].getNormalVector3fMap().cast<double>(),
                                         cl->points[c].intensity));
    }
    cloud->height = cl->height;
    cloud->width = cl->width;
    const Cloud<PType>::Ptr cloud_tf = transform<PType, double>(cloud, in.transform);

    LOG(INFO) << "Compute point projection in camera image";
    const Eigen::Matrix3Xd pts_3d_tf{cloud_tf->getMatrixPoints()};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, cl->size())};
    std::vector<bool> in_img{camera_->getImagePoints(pts_3d_tf, img_pts_raw)};

    int cols, rows;
    camera_->getImageSize(cols, rows);
    LOG(INFO) << "Image size: " << cols << " x " << rows;
    std::map<Pixel, PType, PixelLess> projection, projection_tf;
    for (size_t c = 0; c < in_img.size(); c++) {
        Pixel p(img_pts_raw(0, c), img_pts_raw(1, c));
        if (in_img[c] && (p.row > 0) && (p.row < rows) && (p.col > 0) && (p.col < cols)) {
            p.val = img.at<double>(p.row, p.col);
            projection.insert(std::make_pair(p, cloud->points[c]));
            projection_tf.insert(std::make_pair(p, cloud_tf->points[c]));
        }
    }
    LOG(INFO) << "Number of projections: " << projection.size();

    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Zero(rows, cols)};
    getDepthEst(depth_est, certainty, projection_tf, camera_, params_.initialization,
                params_.neighbor_search);

    LOG(INFO) << "Initialize normals";
    const Cloud<PType>::Ptr cloud_est{Cloud<PType>::create(rows, cols)};
    getNormalEst(*cloud_est, projection, camera_);

    LOG(INFO) << "Create optimization problem";
    std::vector<FunctorDistance::Ptr> functors_distance;
    functors_distance.reserve(projection.size());
    std::vector<FunctorNormal::Ptr> functor_normal;
    functor_normal.reserve(projection.size());
    Eigen::Quaterniond rotation{in.transform.rotation()};
    Eigen::Vector3d translation{in.transform.translation()};
    ceres::Problem problem(params_.problem);
    problem.AddParameterBlock(rotation.coeffs().data(), FunctorDistance::DimRotation,
                              new util_ceres::EigenQuaternionParameterization);
    problem.AddParameterBlock(translation.data(), FunctorDistance::DimTranslation);

    if (params_.use_functor_normal || params_.use_functor_normal_distance ||
        params_.use_functor_smoothness_normal) {
        LOG(INFO) << "Add normal parameterization";
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                problem.AddParameterBlock(
                    cloud_est->at(col, row).normal.data(), FunctorNormal::DimNormal,
                    new util_ceres::ConstantLengthParameterization<FunctorNormal::DimNormal>);
            }
        }
    }

    if (params_.use_functor_distance) {
        LOG(INFO) << "Add distance costs";
        for (auto const& el : projection) {
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(el.first.x, el.first.y), support, direction);
            functors_distance.emplace_back(
                FunctorDistance::create(el.second.position, params_.kd, support, direction));
            problem.AddResidualBlock(functors_distance.back()->toCeres(),
                                     params_.loss_function.get(),
                                     &depth_est(el.first.row, el.first.col),
                                     rotation.coeffs().data(), translation.data());
        }
    }

    if (params_.use_functor_normal) {
        LOG(INFO) << "Add normal costs";
        for (auto const& el : projection) {
            if (params_.use_functor_normal) {
                functor_normal.emplace_back(FunctorNormal::create(el.second.normal, params_.kd));
                problem.AddResidualBlock(functor_normal.back()->toCeres(),
                                         params_.loss_function.get(),
                                         cloud_est->at(el.first.col, el.first.row).normal.data(),
                                         rotation.coeffs().data());
            }
        }
    }

    if (params_.use_functor_smoothness_normal) {
        LOG(INFO) << "Add normal smoothness costs";
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                const Pixel p(col, row, img.at<double>(row, col));
                const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
                for (auto const& n : neighbors) {
                    problem.AddResidualBlock(
                        FunctorSmoothnessNormal::create(
                            smoothnessWeight(p, n, params_.discontinuity_threshold,
                                             params_.smoothness_rate) *
                            params_.ks),
                        new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                              ceres::TAKE_OWNERSHIP),
                        cloud_est->at(col, row).normal.data(),
                        cloud_est->at(n.col, n.row).normal.data());
                }
            }
        }
    }

    if (params_.use_functor_smoothness_distance) {
        LOG(INFO) << "Add distance smoothness costs";
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                const Pixel p(col, row, img.at<double>(row, col));
                const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
                for (auto const& n : neighbors) {
                    problem.AddResidualBlock(
                        FunctorSmoothnessDistance::create(
                            smoothnessWeight(p, n, params_.discontinuity_threshold,
                                             params_.smoothness_rate) *
                            params_.ks),
                        new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                              ceres::TAKE_OWNERSHIP),
                        &depth_est(row, col), &depth_est(n.row, n.col));
                }
            }
        }
    }

    if (params_.use_functor_normal_distance) {
        LOG(INFO) << "Add normal distance cost";
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                const Pixel p(col, row);
                Eigen::Vector3d support_this, direction_this;
                camera_->getViewingRay(Eigen::Vector2d(p.x, p.y), support_this, direction_this);
                const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
                for (auto const& n : neighbors) {
                    Eigen::Vector3d support_nn, direction_nn;
                    camera_->getViewingRay(Eigen::Vector2d(n.x, n.y), support_nn, direction_nn);
                    problem.AddResidualBlock(
                        FunctorNormalDistance::create(
                            params_.kn,
                            Eigen::ParametrizedLine<double, 3>(support_this, direction_this),
                            Eigen::ParametrizedLine<double, 3>(support_nn, direction_nn)),
                        new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                              ceres::TAKE_OWNERSHIP),
                        &depth_est(row, col), &depth_est(n.row, n.col),
                        cloud_est->at(col, row).normal.data());
                }
            }
        }
    }

    if (params_.limits != Parameters::Limits::none) {
        double ub{0}, lb{0};
        if (params_.limits == Parameters::Limits::custom) {
            ub = params_.custom_depth_limit_max;
            lb = params_.custom_depth_limit_min;
            LOG(INFO) << "Use custom limits. Min: " << lb << ", Max: " << ub;
        } else if (params_.limits == Parameters::Limits::adaptive) {
            ub = pts_3d_tf.colwise().norm().maxCoeff();
            lb = pts_3d_tf.colwise().norm().minCoeff();
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
    if (params_.pin_normals) {
        LOG(INFO) << "Pin normals";
        for (auto const& el : projection) {
            problem.SetParameterBlockConstant(&depth_est(el.first.row, el.first.col));
        }
    }
    if (params_.pin_distances) {
        LOG(INFO) << "Pin distances";
        for (auto const& el : projection) {
            problem.SetParameterBlockConstant(
                cloud_est->at(el.first.col, el.first.row).normal.data());
        }
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
    ceres::Solver::Options opt;

    params_.solver.max_solver_time_in_seconds = 120;
    ceres::Solve(params_.solver, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    LOG(INFO) << "Write output data";
    out.transform = util_ceres::fromQuaternionTranslation(rotation, translation);
    cv::eigen2cv(depth_est, out.image);
    out.cloud->reserve(rows * cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(col, row), support, direction);
            PointT p;
            p.getVector3fMap() = (support + direction * depth_est(row, col)).cast<float>();
            p.getNormalVector3fMap() = cloud_est->at(col, row).normal.cast<float>();
            out.cloud->push_back(p);
        }
    }
    out.cloud->width = cols;
    out.cloud->height = rows;
    pcl::transformPointCloudWithNormals(*out.cloud, *out.cloud, out.transform.inverse());

    LOG(INFO) << "Write info";
    ResultInfo info;
    info.optimization_successful = summary.IsSolutionUsable();
    info.number_of_3d_points = projection.size();
    info.number_of_image_points = rows * cols;
    return info;
}
}
