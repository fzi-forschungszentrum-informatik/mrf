#include <chrono>
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

#include "../cloud_preprocessing.hpp"
#include "../cv_helper.hpp"
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

    cv::Mat image{norm_color(in.image, true)};
    LOG(INFO) << "Image size: " << d_.image.cols << " x " << d_.image.rows << " = "
              << d_.image.cols * d_.image.rows;

    LOG(INFO) << "Preprocess and transform cloud";
    pcl::copyPointCloud<T, PointT>(*(in.cloud), *d_.cloud);
    LOG(INFO) << "Cloud size: " << d_.cloud->height << " x " << d_.cloud->width << " = "
              << d_.cloud->size();

    if (params_.estimate_normals) {
        d_.cloud->height = 1; /// < Make cloud unorganized to suppress warnings
        d_.cloud = estimateNormals<PointT, PointT>(d_.cloud, params_.radius_normal_estimation);
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

    LOG(INFO) << "Compute point projections in camera image";
    const Eigen::Matrix3Xd pts_3d_tf{cloud_tf->getMatrixPoints()};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, cloud->size())};
    const std::vector<bool> in_front{camera_->getImagePoints(pts_3d_tf, img_pts_raw)};

    int cols, rows;
    camera_->getImageSize(cols, rows);
    LOG(INFO) << "Image size: " << cols << " x " << rows << " = " << cols * rows;
    std::map<Pixel, PType, PixelLess> projection, projection_tf;
    for (size_t c = 0; c < in_front.size(); c++) {
        const Pixel p(img_pts_raw(0, c), img_pts_raw(1, c));
        if (in_front[c] && p.inImage(rows, cols)) {
            projection.emplace(p, cloud->points[c]);
            projection_tf.emplace(p, cloud_tf->points[c]);
        }
    }
    LOG(INFO) << "Number of projections: " << projection.size();

    int row_min{0}, row_max{rows - 1};
    int col_min{0}, col_max{cols - 1};
    if (params_.crop_mode == Parameters::CropMode::min_max) {
        LOG(INFO) << "Perform 'min_max' box cropping";
        row_min = rows;
        row_max = 0;
        col_min = cols;
        col_max = 0;
        for (auto const& el : projection) {
            const Pixel& p{el.first};
            if (p.col < col_min)
                col_min = p.col;
            else if (p.col > col_max)
                col_max = p.col;
            if (p.row < row_min)
                row_min = p.row;
            else if (p.row > row_max)
                row_max = p.row;
        }
        LOG(INFO) << "row_min: " << row_min << ", row_max: " << row_max << ", col_min: " << col_min
                  << ", col_max: " << col_max;
    }

    LOG(INFO) << "Create ray map";
    std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess> rays;
    for (int row = row_min; row < row_max + 1; row++) {
        for (int col = col_min; col < col_max + 1; col++) {
            const Pixel p(col, row, getVector<float>(image, row, col));
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(p.x, p.y), support, direction);
            rays.emplace(p, Eigen::ParametrizedLine<double, 3>(support, direction));
        }
    }

    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();
    LOG(INFO) << "Estimate prior";
    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Zero(rows, cols)};
    const ClType::Ptr cloud_est{ClType::create(rows, cols)};
    const bool use_any_normals{params_.use_functor_normal || params_.use_functor_normal_distance ||
                               params_.use_functor_smoothness_normal};
    estimatePrior(rays, projection_tf, rows, cols, params_, depth_est, certainty, cloud_est);
    const std::chrono::duration<double> t_diff_prior = Clock::now() - start;

    LOG(INFO) << "Create optimization problem";
    Eigen::Quaterniond rotation{in.transform.rotation()};
    Eigen::Vector3d translation{in.transform.translation()};
    using namespace ceres;
    Problem problem(params_.problem);
    problem.AddParameterBlock(rotation.coeffs().data(),
                              FunctorDistance::DimRotation,
                              new util_ceres::EigenQuaternionParameterization);
    problem.AddParameterBlock(translation.data(), FunctorDistance::DimTranslation);

    LOG(INFO) << "Add parameter blocks";
    for (auto const& el : rays) {
        problem.AddParameterBlock(&depth_est(el.first.row, el.first.col),
                                  FunctorDistance::DimDistance);
        if (use_any_normals)
            problem.AddParameterBlock(
                cloud_est->at(el.first.col, el.first.row).normal.data(),
                FunctorNormal::DimNormal,
                new util_ceres::ConstantLengthParameterization<FunctorNormal::DimNormal>);
    }

    std::vector<ResidualBlockId> ids_functor_distance, ids_functor_normal;
    ids_functor_distance.reserve(projection.size());
    ids_functor_normal.reserve(projection.size());
    for (auto const& el : projection) {
        if (params_.use_functor_distance && !params_.pin_distances)
            ids_functor_distance.emplace_back(problem.AddResidualBlock(
                FunctorDistance::create(el.second.position, rays.at(el.first)),
                new ScaledLoss(params_.loss_function.get(), params_.kd, DO_NOT_TAKE_OWNERSHIP),
                &depth_est(el.first.row, el.first.col),
                rotation.coeffs().data(),
                translation.data()));
        if (params_.use_functor_normal && !params_.pin_normals)
            ids_functor_normal.emplace_back(problem.AddResidualBlock(
                FunctorNormal::create(el.second.normal),
                new ScaledLoss(params_.loss_function.get(), params_.kd, DO_NOT_TAKE_OWNERSHIP),
                cloud_est->at(el.first.col, el.first.row).normal.data(),
                rotation.coeffs().data()));
    }

    std::vector<ResidualBlockId> ids_functor_smoothness_normal, ids_functor_smoothness_distance,
        ids_functor_normal_distance;
    ids_functor_smoothness_normal.reserve(8 * rays.size());
    ids_functor_smoothness_distance.reserve(8 * rays.size());
    ids_functor_normal_distance.reserve(8 * rays.size());
    for (auto const& el : rays) {
        const std::vector<Pixel> neighbors{getNeighbors(el.first, image, params_.neighborhood)};
        for (auto const& n : neighbors) {
            const double w{smoothnessWeight(el.first,
                                            n,
                                            params_.discontinuity_threshold,
                                            params_.smoothness_weight_min,
                                            params_.smoothness_weighting,
                                            params_.smoothness_rate) *
                           params_.ks / neighbors.size()};
            if (params_.use_functor_smoothness_normal)
                ids_functor_smoothness_normal.emplace_back(problem.AddResidualBlock(
                    FunctorSmoothnessNormal::create(),
                    new ScaledLoss(new TrivialLoss, w, TAKE_OWNERSHIP),
                    cloud_est->at(el.first.col, el.first.row).normal.data(),
                    cloud_est->at(n.col, n.row).normal.data()));

            if (params_.use_functor_smoothness_distance)
                ids_functor_smoothness_distance.emplace_back(
                    problem.AddResidualBlock(FunctorSmoothnessDistance::create(),
                                             new ScaledLoss(new TrivialLoss, w, TAKE_OWNERSHIP),
                                             &depth_est(el.first.row, el.first.col),
                                             &depth_est(n.row, n.col)));

            if (params_.use_functor_normal_distance)
                ids_functor_normal_distance.emplace_back(problem.AddResidualBlock(
                    FunctorNormalDistance::create(rays.at(el.first), rays.at(n)),
                    new ScaledLoss(params_.loss_function.get(), w * params_.kn, TAKE_OWNERSHIP),
                    &depth_est(el.first.row, el.first.col),
                    &depth_est(n.row, n.col),
                    cloud_est->at(el.first.col, el.first.row).normal.data()));
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
    if (!params_.solver.IsValid(&err_str))
        LOG(ERROR) << err_str;

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

    cv::Mat img_intensity;
    if (in.image.channels() > 1)
        cv::cvtColor(in.image, img_intensity, CV_BGR2GRAY);
    else
        img_intensity = in.image;

    for (auto const& el : rays) {
        const Pixel& p{el.first};
        out.cloud->at(p.col, p.row).getVector3fMap() =
            el.second.pointAt(depth_est(p.row, p.col)).cast<float>();
        out.cloud->at(p.col, p.row).getNormalVector3fMap() =
            cloud_est->at(p.col, p.row).normal.cast<float>();
        out.cloud->at(p.col, p.row).intensity = img_intensity.at<float>(p.row, p.col);
    }
    pcl::transformPointCloudWithNormals(*out.cloud, *out.cloud, out.transform.inverse());

    LOG(INFO) << "Create result info";
    ResultInfo info;
    info.t_prior = t_diff_prior.count();
    info.t_solver = summary.total_time_in_seconds;
    info.optimization_successful = summary.IsSolutionUsable();
    info.number_of_3d_points = projection.size();
    info.number_of_image_points = rays.size();
    info.iterations_used = summary.iterations.size();

    if (params_.estimate_covariances) {
        LOG(INFO) << "Estimate covariances";
        Covariance::Options options;
        options.num_threads = params_.solver.num_threads;
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
            covariance.GetCovarianceBlock(&depth_est(p.row, p.col),
                                          &depth_est(p.row, p.col),
                                          &(info.covariance_depth(p.row, p.col)));
        }
        info.has_covariance_depth = true;

        if (params_.use_covariance_filter) {
            LOG(INFO) << "Remove points with high covariances";
            std::vector<int> indices_to_keep;
            double cov_max{-std::numeric_limits<double>::max()};
            double cov_min{std::numeric_limits<double>::max()};
            int points_removed{0};
            for (size_t r = 0; r < out.cloud->height; r++) {
                for (size_t c = 0; c < out.cloud->width; c++) {
                    if (info.covariance_depth(r, c) > cov_max)
                        cov_max = info.covariance_depth(r, c);
                    else if (info.covariance_depth(r, c) < cov_min)
                        cov_min = info.covariance_depth(r, c);
                    if (info.covariance_depth(r, c) > params_.covariance_filter_treshold) {
                        out.cloud->at(c, r).x = std::numeric_limits<float>::quiet_NaN();
                        out.cloud->at(c, r).y = std::numeric_limits<float>::quiet_NaN();
                        out.cloud->at(c, r).z = std::numeric_limits<float>::quiet_NaN();
                        points_removed++;
                    }
                }
            }
            LOG(INFO) << "Points removed: " << points_removed;
            LOG(INFO) << "cov_min: " << cov_min << ", cov_max: " << cov_max;
        }
    }
    return info;
}
}
