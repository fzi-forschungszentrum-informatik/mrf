#include <limits>
#include <map>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <pcl_ceres/point.hpp>
#include <pcl_ceres/point_cloud.hpp>
#include <pcl_ceres/transforms.hpp>
#include <util_ceres/constant_length_parameterization.h>
#include <util_ceres/eigen_quaternion_parameterization.h>
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

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
    d_.image = edge(in.image);

    LOG(INFO) << "Preprocess and transform cloud";
    pcl::copyPointCloud<T, PointT>(*(in.cloud), *d_.cloud);

    if (params_.estimate_normals) {
        d_.cloud = estimateNormals<PointT, PointT>(d_.cloud, params_.radius_normal_estimation);
        std::vector<int> indices;
        pcl::removeNaNNormalsFromPointCloud(*d_.cloud, *d_.cloud, indices);
        LOG(INFO) << "Removed " << in.cloud->size() - d_.cloud->size() << " invalid normal points";
    }

    using PType = pcl_ceres::Point<double>;
    using ClType = pcl_ceres::PointCloud<PType>;
    const ClType::Ptr cloud{ClType::create()};
    cloud->points.reserve(d_.cloud->size());
    for (size_t c = 0; c < d_.cloud->size(); c++) {
        cloud->emplace_back(PType(d_.cloud->points[c].getVector3fMap().cast<double>(),
                                  d_.cloud->points[c].getNormalVector3fMap().cast<double>(),
                                  d_.cloud->points[c].intensity));
    }
    const ClType::Ptr cloud_tf = pcl_ceres::transform<PType>(cloud, in.transform);

    LOG(INFO) << "Compute point projection in camera image";
    const Eigen::Matrix3Xd pts_3d_tf{cloud_tf->getMatrixPoints()};

    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, cloud->size())};
    const std::vector<bool> in_img{camera_->getImagePoints(pts_3d_tf, img_pts_raw)};

    int cols, rows;
    camera_->getImageSize(cols, rows);
    LOG(INFO) << "Image size: " << cols << " x " << rows << " = " << cols * rows;
    std::map<Pixel, PType, PixelLess> projection, projection_tf;
    for (size_t c = 0; c < in_img.size(); c++) {
        Pixel p(img_pts_raw(0, c), img_pts_raw(1, c));
        if (in_img[c] && (p.row > 0) && (p.row < rows) && (p.col > 0) && (p.col < cols)) {
            projection.insert(std::make_pair(p, cloud->points[c]));
            projection_tf.insert(std::make_pair(p, cloud_tf->points[c]));
        }
    }
    LOG(INFO) << "Number of projections: " << projection.size();

    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Zero(rows, cols)};
    getDepthEst(depth_est, certainty, projection_tf, camera_, params_.initialization,
                params_.neighbor_search);

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

    std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess> rays;
    LOG(INFO) << "Create ray map";
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Eigen::Vector3d support, direction;
            const Pixel p(col, row, d_.image.at<double>(row, col));
            camera_->getViewingRay(Eigen::Vector2d(p.x, p.y), support, direction);
            rays.insert(std::make_pair(p, Eigen::ParametrizedLine<double, 3>(support, direction)));
        }
    }

    const ClType::Ptr cloud_est{ClType::create(rows, cols)};
    const bool use_any_normals{params_.use_functor_normal || params_.use_functor_normal_distance ||
                               params_.use_functor_smoothness_normal};
    if (use_any_normals) {
        LOG(INFO) << "Initialize normals";
        getNormalEst(*cloud_est, projection, camera_);
        LOG(INFO) << "Add normal parameterization";
        for (auto const& el : rays) {
            problem.AddParameterBlock(
                cloud_est->at(el.first.col, el.first.row).normal.data(), FunctorNormal::DimNormal,
                new util_ceres::ConstantLengthParameterization<FunctorNormal::DimNormal>);
        }
    }

    if (params_.use_functor_distance) {
        LOG(INFO) << "Add distance costs";
        for (auto const& el : projection) {
            functors_distance.emplace_back(
                FunctorDistance::create(el.second.position, rays.at(el.first)));
            problem.AddResidualBlock(functors_distance.back()->toCeres(),
                                     new ceres::ScaledLoss(params_.loss_function.get(), params_.kd,
                                                           ceres::DO_NOT_TAKE_OWNERSHIP),
                                     &depth_est(el.first.row, el.first.col),
                                     rotation.coeffs().data(), translation.data());
        }
    }

    if (params_.use_functor_normal) {
        LOG(INFO) << "Add normal costs";
        for (auto const& el : projection) {
            if (params_.use_functor_normal) {
                functor_normal.emplace_back(FunctorNormal::create(el.second.normal));
                problem.AddResidualBlock(functor_normal.back()->toCeres(),
                                         new ceres::ScaledLoss(params_.loss_function.get(),
                                                               params_.kd,
                                                               ceres::DO_NOT_TAKE_OWNERSHIP),
                                         cloud_est->at(el.first.col, el.first.row).normal.data(),
                                         rotation.coeffs().data());
            }
        }
    }

    if (params_.use_functor_smoothness_normal) {
        LOG(INFO) << "Add normal smoothness costs";
        for (auto const& el : rays) {
            const std::vector<Pixel> neighbors{
                getNeighbors(el.first, d_.image, params_.neighborhood)};
            for (auto const& n : neighbors) {
                const double w{smoothnessWeight(el.first, n, params_.discontinuity_threshold,
                                                params_.smoothness_rate) *
                               params_.ks / neighbors.size()};
                problem.AddResidualBlock(
                    FunctorSmoothnessNormal::create(),
                    new ceres::ScaledLoss(new ceres::TrivialLoss, w, ceres::TAKE_OWNERSHIP),
                    cloud_est->at(el.first.col, el.first.row).normal.data(),
                    cloud_est->at(n.col, n.row).normal.data());
            }
        }
    }

    if (params_.use_functor_smoothness_distance) {
        LOG(INFO) << "Add distance smoothness costs";
        for (auto const& el : rays) {
            const std::vector<Pixel> neighbors{
                getNeighbors(el.first, d_.image, params_.neighborhood)};
            for (auto const& n : neighbors) {
                const double w{smoothnessWeight(el.first, n, params_.discontinuity_threshold,
                                                params_.smoothness_rate) *
                               params_.ks / neighbors.size()};
                problem.AddResidualBlock(
                    FunctorSmoothnessDistance::create(),
                    new ceres::ScaledLoss(new ceres::TrivialLoss, w, ceres::TAKE_OWNERSHIP),
                    &depth_est(el.first.row, el.first.col), &depth_est(n.row, n.col));
            }
        }
    }

    if (params_.use_functor_normal_distance) {
        LOG(INFO) << "Add normal distance cost";
        for (auto const& el : rays) {
            const std::vector<Pixel> neighbors{
                getNeighbors(el.first, d_.image, params_.neighborhood)};
            for (auto const& n : neighbors) {
                problem.AddResidualBlock(
                    FunctorNormalDistance::create(rays.at(el.first), rays.at(n)),
                    new ceres::ScaledLoss(new ceres::TrivialLoss, params_.kn / neighbors.size(),
                                          ceres::TAKE_OWNERSHIP),
                    &depth_est(el.first.row, el.first.col), &depth_est(n.row, n.col),
                    cloud_est->at(el.first.col, el.first.row).normal.data());
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
        for (auto const& el : rays) {
            problem.SetParameterLowerBound(&depth_est(el.first.row, el.first.col), 0, lb);
            problem.SetParameterUpperBound(&depth_est(el.first.row, el.first.col), 0, ub);
        }
    }

    if (pin_transform) {
        LOG(INFO) << "Pin transform";
        problem.SetParameterBlockConstant(rotation.coeffs().data());
        problem.SetParameterBlockConstant(translation.data());
    }
    if (params_.pin_normals && use_any_normals) {
        LOG(INFO) << "Pin normals";
        for (auto const& el : projection) {
            problem.SetParameterBlockConstant(&depth_est(el.first.row, el.first.col));
        }
    }
    const bool use_any_distances{params_.use_functor_distance ||
                                 params_.use_functor_smoothness_distance ||
                                 params_.use_functor_normal_distance};
    if (params_.pin_distances && use_any_distances) {
        LOG(INFO) << "Pin distances";
        for (auto const& el : projection) {
            problem.SetParameterBlockConstant(&depth_est(el.first.row, el.first.col));
        }
    }

    std::string err_str;
    if (!params_.solver.IsValid(&err_str)) {
        LOG(ERROR) << err_str;
    }

    LOG(INFO) << "Solve problem";
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solve(params_.solver, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    LOG(INFO) << "Write output data";
    out.transform = util_ceres::fromQuaternionTranslation(rotation, translation);
    cv::eigen2cv(depth_est, out.image);
    out.cloud->width = cols;
    out.cloud->height = rows;
    out.cloud->resize(rays.size());
    for (auto const& el : rays) {
        out.cloud->at(el.first.col, el.first.row).getVector3fMap() =
            el.second.pointAt(depth_est(el.first.row, el.first.col)).cast<float>();
        out.cloud->at(el.first.col, el.first.row).getNormalVector3fMap() =
            cloud_est->at(el.first.col, el.first.row).normal.cast<float>();
    }

    pcl::transformPointCloudWithNormals(*out.cloud, *out.cloud, out.transform.inverse());

    LOG(INFO) << "Write info";
    ResultInfo info;
    info.optimization_successful = summary.IsSolutionUsable();
    info.number_of_3d_points = projection.size();
    info.number_of_image_points = rows * cols;
    return info;
}
}
