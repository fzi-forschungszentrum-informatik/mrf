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
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

#include "../cloud_preprocessing.hpp"
#include "../functor_distance.hpp"
#include "../functor_normal.hpp"
#include "../functor_normal_distance.hpp"
#include "../functor_smoothness_distance.hpp"
#include "../functor_smoothness_normal.hpp"
#include "../image_preprocessing.hpp"
#include "../neighbors.hpp"
#include "../prior.hpp"
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

    LOG(INFO) << "Create ray map";
    std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess> rays;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Eigen::Vector3d support, direction;
            const Pixel p(col, row, d_.image.at<double>(row, col));
            camera_->getViewingRay(Eigen::Vector2d(p.x, p.y), support, direction);
            rays.insert(std::make_pair(p, Eigen::ParametrizedLine<double, 3>(support, direction)));
        }
    }

    LOG(INFO) << "Estimate initial depths";
    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Zero(rows, cols)};
    const ClType::Ptr cloud_est{ClType::create(rows, cols)};
    const bool use_any_normals{params_.use_functor_normal || params_.use_functor_normal_distance ||
                               params_.use_functor_smoothness_normal};

    LOG(INFO) << "Estimate priors";
    estimatePrior(rays, projection_tf, rows, cols, params_.initialization, params_.neighbor_search,
                  depth_est, certainty, cloud_est);

    LOG(INFO) << "Create optimization problem";
    Eigen::Quaterniond rotation{in.transform.rotation()};
    Eigen::Vector3d translation{in.transform.translation()};
    using namespace ceres;
    Problem problem(params_.problem);
    problem.AddParameterBlock(rotation.coeffs().data(), FunctorDistance::DimRotation,
                              new util_ceres::EigenQuaternionParameterization);
    problem.AddParameterBlock(translation.data(), FunctorDistance::DimTranslation);

    LOG(INFO) << "Adding parameter blocks";
    for (auto const& el : rays) {
        problem.AddParameterBlock(&depth_est(el.first.row, el.first.col),
                                  FunctorDistance::DimDistance);
        if (use_any_normals) {
            problem.AddParameterBlock(
                cloud_est->at(el.first.col, el.first.row).normal.data(), FunctorNormal::DimNormal,
                new util_ceres::ConstantLengthParameterization<FunctorNormal::DimNormal>);
        }
    }

    std::vector<ResidualBlockId> ids_functor_distance, ids_functor_normal;
    ids_functor_distance.reserve(projection.size());
    ids_functor_normal.reserve(projection.size());
    for (auto const& el : projection) {
        if (params_.use_functor_distance) {
            ids_functor_distance.emplace_back(problem.AddResidualBlock(
                FunctorDistance::create(el.second.position, rays.at(el.first)),
                new ScaledLoss(params_.loss_function.get(), params_.kd, DO_NOT_TAKE_OWNERSHIP),
                &depth_est(el.first.row, el.first.col), rotation.coeffs().data(),
                translation.data()));
        }
        if (params_.use_functor_normal) {
            ids_functor_normal.emplace_back(problem.AddResidualBlock(
                FunctorNormal::create(el.second.normal),
                new ScaledLoss(params_.loss_function.get(), params_.kd, DO_NOT_TAKE_OWNERSHIP),
                cloud_est->at(el.first.col, el.first.row).normal.data(), rotation.coeffs().data()));
        }
    }

    std::vector<ResidualBlockId> ids_functor_smoothness_normal, ids_functor_smoothness_distance,
        ids_functor_normal_distance;
    ids_functor_smoothness_normal.reserve(rays.size());
    ids_functor_smoothness_distance.reserve(rays.size());
    ids_functor_normal_distance.reserve(rays.size());
    for (auto const& el : rays) {
        const std::vector<Pixel> neighbors{getNeighbors(el.first, d_.image, params_.neighborhood)};
        for (auto const& n : neighbors) {
            const double w{smoothnessWeight(el.first, n, params_.discontinuity_threshold,
                                            params_.smoothness_rate,
                                            params_.smoothness_weight_min) *
                           params_.ks / neighbors.size()};
            if (params_.use_functor_smoothness_normal) {
                ids_functor_smoothness_normal.emplace_back(problem.AddResidualBlock(
                    FunctorSmoothnessNormal::create(),
                    new ScaledLoss(new TrivialLoss, w, TAKE_OWNERSHIP),
                    cloud_est->at(el.first.col, el.first.row).normal.data(),
                    cloud_est->at(n.col, n.row).normal.data()));
            }
            if (params_.use_functor_smoothness_distance) {
                ids_functor_smoothness_distance.emplace_back(problem.AddResidualBlock(
                    FunctorSmoothnessDistance::create(),
                    new ScaledLoss(new TrivialLoss, w, TAKE_OWNERSHIP),
                    &depth_est(el.first.row, el.first.col), &depth_est(n.row, n.col)));
            }
            if (params_.use_functor_normal_distance) {
                ids_functor_normal_distance.emplace_back(problem.AddResidualBlock(
                    FunctorNormalDistance::create(rays.at(el.first), rays.at(n)),
                    new ScaledLoss(params_.loss_function.get(), w, TAKE_OWNERSHIP),
                    &depth_est(el.first.row, el.first.col), &depth_est(n.row, n.col),
                    cloud_est->at(el.first.col, el.first.row).normal.data()));
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
            ub = depth_est.maxCoeff();
            lb = depth_est.minCoeff();
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
    if (use_any_normals && params_.pin_normals) {
        LOG(INFO) << "Pin normals";
        for (auto const& el : projection) {
            problem.SetParameterBlockConstant(&depth_est(el.first.row, el.first.col));
        }
    }
    const bool use_any_distances{params_.use_functor_distance ||
                                 params_.use_functor_smoothness_distance ||
                                 params_.use_functor_normal_distance};
    if (use_any_distances && params_.pin_distances) {
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
    Solve(params_.solver, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    LOG(INFO) << "Write output data";
    out.transform = util_ceres::fromQuaternionTranslation(rotation, translation);
    cv::eigen2cv(depth_est, out.image);
    out.cloud->width = cols;
    out.cloud->height = rows;
    out.cloud->resize(out.cloud->width * out.cloud->height);
    for (auto const& el : rays) {
        out.cloud->at(el.first.col, el.first.row).getVector3fMap() =
            el.second.pointAt(depth_est(el.first.row, el.first.col)).cast<float>();
        out.cloud->at(el.first.col, el.first.row).getNormalVector3fMap() =
            cloud_est->at(el.first.col, el.first.row).normal.cast<float>();

        if (in.image.channels() > 1) {
            cv::Mat img;
            cv::cvtColor(in.image, img, CV_BGR2GRAY);
            out.cloud->at(el.first.col, el.first.row).intensity =
                img.at<float>(el.first.row, el.first.col);
        } else {
            out.cloud->at(el.first.col, el.first.row).intensity =
                in.image.template at<float>(el.first.row, el.first.col);
        }
    }
    pcl::transformPointCloudWithNormals(*out.cloud, *out.cloud, out.transform.inverse());

    LOG(INFO) << "Create result info";
    ResultInfo info;
    info.optimization_successful = summary.IsSolutionUsable();
    info.number_of_3d_points = projection.size();
    info.number_of_image_points = rays.size();

    if (params_.estimate_covariances) {
        LOG(INFO) << "Estimate covariances";
        Covariance::Options options;
        Covariance covariance(options);
        std::vector<std::pair<const double*, const double*>> covariance_blocks;
        covariance_blocks.reserve(rays.size());
        for (auto const& el : rays) {
            covariance_blocks.emplace_back(std::make_pair(&depth_est(el.first.row, el.first.col),
                                                          &depth_est(el.first.row, el.first.col)));
        }
        CHECK(covariance.Compute(covariance_blocks, &problem));
        info.covariance_depth.resize(rows, cols);
        for (auto const& el : rays) {
            auto const& p = el.first;
            covariance.GetCovarianceBlock(&depth_est(p.row, p.col), &depth_est(p.row, p.col),
                                          &(info.covariance_depth(p.row, p.col)));
        }
    }

    return info;
}
}
