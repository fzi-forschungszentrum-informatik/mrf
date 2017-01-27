#include <chrono>
#include <map>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <pcl_ceres/point.hpp>
#include <pcl_ceres/point_cloud.hpp>
#include <pcl_ceres/transforms.hpp>
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
#include "../image_preprocessing.hpp"
#include "../neighbors.hpp"
#include "../prior.hpp"
#include "../smoothness_weight.hpp"

namespace mrf {

template <typename T>
ResultInfo Solver::solve(const Data<T>& in, Data<PointT>& out, const bool pin_transform) {

    LOG(INFO) << "Normalize image";
    cv::normalize(in.image, d_.image, 0, 1, cv::NORM_MINMAX);
    LOG(INFO) << "Image size: " << d_.image.cols << " x " << d_.image.rows << " = "
              << d_.image.cols * d_.image.rows;

    LOG(INFO) << "Get Projection Image";
    projection_.image = get_gray_image(in.image);
    cv::cvtColor(projection_.image, projection_.image, CV_GRAY2BGR);

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
                                  d_.cloud->points[c].getNormalVector3fMap().cast<double>()));
    }
    const ClType::Ptr cloud_tf = pcl_ceres::transform<PType>(cloud, in.transform);

    LOG(INFO) << "Compute point projections in camera image";
    const Eigen::Matrix3Xd pts_3d_tf{cloud_tf->getMatrixPoints()};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, cloud->size())};
    const std::vector<bool> in_front{camera_->getImagePoints(pts_3d_tf, img_pts_raw)};
    int cols, rows;
    camera_->getImageSize(cols, rows);
    LOG(INFO) << "Image size: " << cols << " x " << rows << " = " << cols * rows;

    int row_min{0}, row_max{rows};
    int col_min{0}, col_max{cols};
    if (params_.crop_mode == Parameters::CropMode::min_max) {
        LOG(INFO) << "Perform 'min_max' box cropping";
        row_min = rows;
        row_max = 0;
        col_min = cols;
        col_max = 0;
        for (size_t c = 0; c < in_front.size(); c++) {
            const Pixel p(img_pts_raw(0, c), img_pts_raw(1, c));
            if (in_front[c] && p.inImage(rows, cols)) {
                if (p.col < col_min)
                    col_min = p.col;
                else if (p.col > col_max)
                    col_max = p.col;
                if (p.row < row_min)
                    row_min = p.row;
                else if (p.row > row_max)
                    row_max = p.row;
            }
        }
    } else if (params_.crop_mode == Parameters::CropMode::box) {
        if (params_.box_cropping_row_min > 0)
            row_min = params_.box_cropping_row_min;
        if (params_.box_cropping_row_max < row_max)
            row_max = params_.box_cropping_row_max;
        if (params_.box_cropping_col_min > 0)
            col_min = params_.box_cropping_row_min;
        if (params_.box_cropping_row_max < col_max)
            col_max = params_.box_cropping_col_max;
    }
    LOG(INFO) << "row_min: " << row_min << ", row_max: " << row_max << ", col_min: " << col_min
              << ", col_max: " << col_max;


    LOG(INFO) << "Create projection map";
    std::map<Pixel, PType, PixelLess> projection, projection_tf;
    for (size_t c = 0; c < in_front.size(); c++) {
        const Pixel p(img_pts_raw(0, c), img_pts_raw(1, c));
        if (in_front[c] && p.inImage(row_max, col_max, row_min, col_min)) {
            projection.emplace(p, cloud->points[c]);
            projection_tf.emplace(p, cloud_tf->points[c]);
            projection_.image.at<cv::Vec3f>(p.row, p.col)[0] = 0;
            projection_.image.at<cv::Vec3f>(p.row, p.col)[1] = 255;
            projection_.image.at<cv::Vec3f>(p.row, p.col)[2] = 255;
        }
    }
    LOG(INFO) << "Number of projections: " << projection.size();

    LOG(INFO) << "Create ray map";
    std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess> rays;
    for (int row = row_min; row < row_max; row++) {
        for (int col = col_min; col < col_max; col++) {
            const Pixel p(col, row, getVector<float>(d_.image, row, col));
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(p.x, p.y), support, direction);
            rays.emplace(p, Eigen::ParametrizedLine<double, 3>(support, direction));
        }
    }

    using Clock = std::chrono::high_resolution_clock;
    LOG(INFO) << "Estimate prior";
    auto start = Clock::now();
    depth_est_.conservativeResize(rows, cols);
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Zero(rows, cols)};
    estimatePrior(rays, projection_tf, rows, cols, params_, depth_est_, certainty);
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

    LOG(INFO) << "Add " << rays.size() << " parameter blocks";
    for (auto const& el : rays)
        problem.AddParameterBlock(&depth_est_(el.first.row, el.first.col),
                                  FunctorDistance::DimDistance);

    LOG(INFO) << "Add measurement residual blocks";
    std::vector<ResidualBlockId> ids_functor_distance, ids_functor_normal;
    ids_functor_distance.reserve(projection.size());
    ids_functor_normal.reserve(projection.size());
    for (auto const& el_projection : projection) {
        const Pixel& p{el_projection.first};
        if (params_.use_functor_distance && !params_.pin_distances)
            ids_functor_distance.emplace_back(problem.AddResidualBlock(
                FunctorDistance::create(el_projection.second.position, rays.at(p)),
                new ScaledLoss(params_.loss_function.get(), params_.kd, DO_NOT_TAKE_OWNERSHIP),
                &depth_est_(p.row, p.col),
                rotation.coeffs().data(),
                translation.data()));

        if (params_.use_functor_normal)
            for (auto const& n : getNeighbors(
                     p, d_.image, params_.neighborhood, row_max, col_max, row_min, col_min))
                ids_functor_normal.emplace_back(problem.AddResidualBlock(
                    FunctorNormal::create(el_projection.second.normal, rays.at(p), rays.at(n)),
                    new ScaledLoss(params_.loss_function.get(), params_.kd, DO_NOT_TAKE_OWNERSHIP),
                    rotation.coeffs().data(),
                    &depth_est_(p.row, p.col),
                    &depth_est_(n.row, n.col)));
    }

    LOG(INFO) << "Add smoothness residual blocks";
    std::vector<ResidualBlockId> ids_functor_smoothness_distance, ids_functor_normal_distance;
    ids_functor_smoothness_distance.reserve(rays.size());
    ids_functor_normal_distance.reserve(2 * rays.size());
    Eigen::MatrixXd weights{Eigen::MatrixXd::Zero(rows, cols)};
    for (auto const& el_ray : rays) {
        const Pixel& p{el_ray.first};
        const std::vector<Pixel> neighbors{
            getNeighbors(p, d_.image, params_.neighborhood, row_max, col_max, row_min, col_min)};
        std::vector<double> weights;
        for (auto const& n : neighbors)
            weights.emplace_back(smoothnessWeight(p,
                                                  n,
                                                  params_.discontinuity_threshold,
                                                  params_.smoothness_weight_min,
                                                  params_.smoothness_weighting,
                                                  params_.smoothness_rate,
                                                  1));

        if (params_.use_functor_normal_distance) {
            if (neighbors.size() > 2)
                ids_functor_normal_distance.emplace_back(problem.AddResidualBlock(
                    FunctorNormalDistance::create(rays.at(p),
                                                  rays.at(neighbors[0]),
                                                  rays.at(neighbors[2])),
                    new ScaledLoss(params_.loss_function.get(), params_.ks * weights[0] * weights[2], DO_NOT_TAKE_OWNERSHIP),
                    &depth_est_(p.row, p.col),
                    &depth_est_(neighbors[0].row, neighbors[0].col),
                    &depth_est_(neighbors[2].row, neighbors[2].col)));
            if (neighbors.size() > 3)
                ids_functor_normal_distance.emplace_back(problem.AddResidualBlock(
                    FunctorNormalDistance::create(rays.at(p),
                                                  rays.at(neighbors[1]),
                                                  rays.at(neighbors[3])),
                    new ScaledLoss(params_.loss_function.get(), params_.ks * weights[1] * weights[3], DO_NOT_TAKE_OWNERSHIP),
                    &depth_est_(p.row, p.col),
                    &depth_est_(neighbors[1].row, neighbors[1].col),
                    &depth_est_(neighbors[3].row, neighbors[3].col)));
        }

        if (params_.use_functor_smoothness_distance)
            for (size_t c = 0; c < neighbors.size(); c++)
                ids_functor_smoothness_distance.emplace_back(problem.AddResidualBlock(
                    FunctorSmoothnessDistance::create(),
                    new ScaledLoss(new TrivialLoss, weights[c], TAKE_OWNERSHIP),
                    &depth_est_(p.row, p.col),
                    &depth_est_(neighbors[c].row, neighbors[c].col)));
    }

    if (params_.limits != Parameters::Limits::none) {
        double ub{0}, lb{0};
        if (params_.limits == Parameters::Limits::custom) {
            ub = params_.custom_depth_limit_max;
            lb = params_.custom_depth_limit_min;
            LOG(INFO) << "Use custom limits. Min: " << lb << ", Max: " << ub;
        } else if (params_.limits == Parameters::Limits::adaptive) {
            ub = depth_est_.maxCoeff();
            lb = depth_est_.minCoeff();
            LOG(INFO) << "Use adaptive limits. Min: " << lb << ", Max: " << ub;
        }
        for (auto const& el : rays) {
            problem.SetParameterLowerBound(&depth_est_(el.first.row, el.first.col), 0, lb);
            problem.SetParameterUpperBound(&depth_est_(el.first.row, el.first.col), 0, ub);
        }
    }

    if (pin_transform) {
        LOG(INFO) << "Pin transform";
        problem.SetParameterBlockConstant(rotation.coeffs().data());
        problem.SetParameterBlockConstant(translation.data());
    }
    const bool use_any_distances{params_.use_functor_distance ||
                                 params_.use_functor_smoothness_distance ||
                                 params_.use_functor_normal_distance};
    if (use_any_distances && params_.pin_distances) {
        LOG(INFO) << "Pin distances";
        for (auto const& el : projection)
            problem.SetParameterBlockConstant(&depth_est_(el.first.row, el.first.col));
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
    cv::eigen2cv(depth_est_, out.image);
    out.cloud->width = cols;
    out.cloud->height = rows;
    out.cloud->resize(out.cloud->width * out.cloud->height);

    cv::Mat img_intensity;
    if (d_.image.channels() > 1)
        cv::cvtColor(d_.image, img_intensity, CV_BGR2GRAY);
    else
        img_intensity = d_.image;

    for (auto const& el : rays) {
        const Pixel& p{el.first};
        const std::vector<Pixel> neighbors{
            getNeighbors(p, d_.image, params_.neighborhood, row_max, col_max, row_min, col_min)};
        weights(p.row, p.col) = smoothnessWeight(p,
                                                 neighbors[0],
                                                 params_.discontinuity_threshold,
                                                 params_.smoothness_weight_min,
                                                 params_.smoothness_weighting,
                                                 params_.smoothness_rate,
                                                 1);

        if (neighbors.size() < 3)
            out.cloud->at(p.col, p.row).getNormalVector3fMap() =
                estimateNormal1(depth_est_(p.row, p.col),
                                rays.at(p),
                                depth_est_(neighbors[0].row, neighbors[0].col),
                                rays.at(neighbors[0]),
                                depth_est_(neighbors[1].row, neighbors[1].col),
                                rays.at(neighbors[1]))
                    .cast<float>();
        else if (neighbors.size() < 4)
            out.cloud->at(p.col, p.row).getNormalVector3fMap() =
                estimateNormal2(depth_est_(p.row, p.col),
                                rays.at(p),
                                depth_est_(neighbors[0].row, neighbors[0].col),
                                rays.at(neighbors[0]),
                                depth_est_(neighbors[1].row, neighbors[1].col),
                                rays.at(neighbors[1]),
                                depth_est_(neighbors[2].row, neighbors[2].col),
                                rays.at(neighbors[2]),
                                0.1)
                    .cast<float>();
        else
            out.cloud->at(p.col, p.row).getNormalVector3fMap() =
                estimateNormal4(depth_est_(p.row, p.col),
                                rays.at(p),
                                depth_est_(neighbors[0].row, neighbors[0].col),
                                rays.at(neighbors[0]),
                                depth_est_(neighbors[1].row, neighbors[1].col),
                                rays.at(neighbors[1]),
                                depth_est_(neighbors[2].row, neighbors[2].col),
                                rays.at(neighbors[2]),
                                depth_est_(neighbors[3].row, neighbors[3].col),
                                rays.at(neighbors[3]),
                                0.1)
                    .cast<float>();

        out.cloud->at(p.col, p.row).getVector3fMap() =
            el.second.pointAt(depth_est_(p.row, p.col)).cast<float>();

        out.cloud->at(p.col, p.row).b = std::numeric_limits<uint8_t>::max() * p.val[0];
        out.cloud->at(p.col, p.row).g = std::numeric_limits<uint8_t>::max() * p.val[1];
        out.cloud->at(p.col, p.row).r = std::numeric_limits<uint8_t>::max() * p.val[2];
        out.cloud->at(p.col, p.row).a = std::numeric_limits<uint8_t>::max();
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
    info.has_weights = true;
    info.weights = weights;

    if (params_.estimate_covariances) {
        LOG(INFO) << "Estimate covariances";
        Covariance::Options options;
        options.num_threads = params_.solver.num_threads;
        Covariance covariance(options);
        std::vector<std::pair<const double*, const double*>> covariance_blocks;
        covariance_blocks.reserve(rays.size());
        for (auto const& el : rays)
            covariance_blocks.emplace_back(std::make_pair(&depth_est_(el.first.row, el.first.col),
                                                          &depth_est_(el.first.row, el.first.col)));

        CHECK(covariance.Compute(covariance_blocks, &problem));
        info.covariance_depth.resize(rows, cols);
        for (auto const& el : rays) {
            auto const& p = el.first;
            covariance.GetCovarianceBlock(&depth_est_(p.row, p.col),
                                          &depth_est_(p.row, p.col),
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
