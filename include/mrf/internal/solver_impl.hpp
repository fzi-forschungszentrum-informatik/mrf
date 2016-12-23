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
#include "../normal_prior.hpp"
#include "../functor_distance.hpp"
#include "../functor_normal.hpp"
#include "../functor_normal_distance.hpp"
#include "../functor_smoothness_distance.hpp"
#include "../functor_smoothness_normal.hpp"
#include "../image_preprocessing.hpp"
#include "../neighbors.hpp"
#include "../smoothness_weight.hpp"

namespace mrf {

template <typename T>
ResultInfo Solver::solve(const Data<T>& in, Data<PointT>& out, const bool pin_transform) {

    LOG(INFO) << "Preprocess image";
    const cv::Mat img{edge(in.image)};

    LOG(INFO) << "Preprocess and transform cloud";
    using CloudT = pcl::PointCloud<PointT>;
    CloudT::Ptr cl{new CloudT};
    CloudT::Ptr cl_tf{new CloudT};
    pcl::copyPointCloud<T, PointT>(*(in.cloud), *cl);
    if (params_.estimate_normals) {
        cl = estimateNormals<PointT, PointT>(cl, params_.radius_normal_estimation);
    }
    std::vector<int> indices;
    pcl::removeNaNNormalsFromPointCloud(*cl, *cl, indices);
    pcl::transformPointCloudWithNormals(*cl, *cl_tf, in.transform);

    LOG(INFO) << "Compute point projection in camera image";
    const Eigen::Matrix3Xd pts_3d_tf{cl_tf->getMatrixXfMap().topRows<3>().cast<double>()};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, cl->size())};
    std::vector<bool> in_img{camera_->getImagePoints(pts_3d_tf, img_pts_raw)};
    int cols, rows;
    camera_->getImageSize(cols, rows);
    LOG(INFO) << "Image size: " << cols << " x " << rows;
    std::map<Pixel, PointT, PixelLess> projection, projection_tf;
    for (size_t c = 0; c < in_img.size(); c++) {
        Pixel p(img_pts_raw(0, c), img_pts_raw(1, c));
        if (in_img[c] && (p.row > 0) && (p.row < rows) && (p.col > 0) && (p.col < cols)) {
            p.val = img.at<double>(p.row, p.col);
            projection.insert(std::make_pair(p, cl->points[c]));
            projection_tf.insert(std::make_pair(p, cl_tf->points[c]));
        }
    }
    LOG(INFO) << "projection size is: " << projection.size();

    Cloud<double> cloud(rows, cols);

    LOG(INFO) << "Create optimization problem";
    ceres::Problem problem(params_.problem);
    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Zero(rows, cols)};
    getDepthEst(depth_est, certainty, projection_tf, camera_, params_.initialization,
                params_.neighbor_search);

    LOG(INFO) << "Initialise Cloud with given normals";
    getNormalEst<double>(cloud,projection,camera_);

    std::vector<FunctorDistance::Ptr> functors_distance;
    functors_distance.reserve(projection.size());
    std::vector<FunctorNormal::Ptr> functor_normal;
    functor_normal.reserve(projection.size());
    Eigen::Quaterniond rotation{in.transform.rotation()};
    Eigen::Vector3d translation{in.transform.translation()};
    problem.AddParameterBlock(rotation.coeffs().data(), FunctorDistance::DimRotation,
                              new util_ceres::EigenQuaternionParameterization);
    problem.AddParameterBlock(translation.data(), FunctorDistance::DimTranslation);

    LOG(INFO) << "Add distance and normal costs";
    for (auto const& el : projection) {
        Eigen::Vector3d support, direction;
        camera_->getViewingRay(Eigen::Vector2d(el.first.x, el.first.y), support, direction);
        //        LOG(INFO) << "Pixel: " << el.first << ", point: " << el.second.transpose()
        //                  << ", support: " << support.transpose()
        //                  << ", direction: " << direction.transpose();
        /**
         * Add Distance Costs
         */
        if (params_.use_functor_distance) {
            functors_distance.emplace_back(FunctorDistance::create(
                el.second.getVector3fMap().cast<double>(), params_.kd, support, direction));
            problem.AddResidualBlock(functors_distance.back()->toCeres(),
                                     params_.loss_function.get(),
                                     &depth_est(el.first.row, el.first.col),
                                     rotation.coeffs().data(), translation.data());
        }
        /**
         * Add Normal Costs
         */
        if (params_.use_functor_normal) {
            functor_normal.emplace_back(
                FunctorNormal::create(el.second.getNormalVector3fMap().cast<double>(), params_.kd));
            problem.AddResidualBlock(functor_normal.back()->toCeres(), params_.loss_function.get(),
                                     cloud.at(el.first.col, el.first.row).normal.data(),
                                     rotation.coeffs().data());
        }
    }

    if (params_.use_functor_smoothness_normal) {
        LOG(INFO) << "Add Normal Smoothness Costs";
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                const Pixel p(col, row, img.at<double>(row, col));
                // LOG(INFO) << "px: "<< normal_x_est(row, col) << ", py: "<<normal_y_est(row,
                // col)<<",
                // nz: "<< normal_z_est(row, col);
                const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
                for (auto const& n : neighbors) {
                    // LOG(INFO) << "nx: "<< normal_x_est(n.row, n.col) << ", ny:
                    // "<<normal_y_est(n.row,
                    // n.col)<<", nz: "<< normal_z_est(n.row, n.col);
                    problem.AddResidualBlock(
                        FunctorSmoothnessNormal::create(smoothnessWeight(p, n, params_) *
                                                        params_.ks),
                        new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                              ceres::TAKE_OWNERSHIP),
                        cloud.at(col, row).normal.data(), cloud.at(n.col, n.row).normal.data());
                }
            }
        }
    }

    if (params_.use_functor_smoothness_distance) {

        LOG(INFO) << "Add smoothness costs";
        std::vector<std::vector<ceres::ResidualBlockId>> smoothness_blocks;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                const Pixel p(col, row, img.at<double>(row, col));
                const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
                std::vector<ceres::ResidualBlockId> res_blocks;
                for (auto const& n : neighbors) {
                    ceres::ResidualBlockId block_id{problem.AddResidualBlock(
                        FunctorSmoothnessDistance::create(smoothnessWeight(p, n, params_) *
                                                          params_.ks),
                        new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                              ceres::TAKE_OWNERSHIP),
                        &depth_est(row, col), &depth_est(n.row, n.col))};
                    res_blocks.push_back(block_id);
                }
                smoothness_blocks.push_back(res_blocks);
            }
        }
    }

    const Eigen::ParametrizedLine<double, 3> ray_nn(Eigen::Vector3<double>(0, 0, 0),
                                                    Eigen::Vector3<double>(0, 0, 1));
    Eigen::Hyperplane<double, 3> plane_this(Eigen::Vector3<double>(0, 1, 0),
                                            Eigen::Vector3<double>(0, 1, 0));
    LOG(INFO) << "intersection of imossible point: " << ray_nn.intersectionPoint(plane_this);

    if (params_.use_functor_normal_distance) {
        std::vector<ceres::ResidualBlockId> normal_dist_ids;
        LOG(INFO) << "Add Normal distance cost";
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                const Pixel p(col, row);
                Eigen::Vector3d support_this, direction_this;
                camera_->getViewingRay(Eigen::Vector2d(p.x, p.y), support_this, direction_this);
                const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
                for (auto const& n : neighbors) {
                    Eigen::Vector3d support_nn, direction_nn;
                    camera_->getViewingRay(Eigen::Vector2d(n.x, n.y), support_nn, direction_nn);
                    // LOG(INFO) << "nx: "<< normal_x_est(n.row, n.col) << ", ny:
                    // "<<normal_y_est(n.row,
                    // n.col)<<", nz: "<< normal_z_est(n.row, n.col);
                    //                    const Eigen::ParametrizedLine<double, 3>
                    //                    ray_nn(support_nn, direction_nn);
                    //                    Eigen::Hyperplane<double, 3> plane_this(
                    //                        Eigen::Vector3<double>(normal_x_est(row, col),
                    //                        normal_y_est(row, col),
                    //                                               normal_z_est(row, col)),
                    //                        support_this + direction_this * depth_est(row, col));
                    //                    LOG(INFO) << "intersection point: " <<
                    //                    ray_nn.intersectionPoint(plane_this);
                    //                    LOG(INFO) << "other: " << ray_nn.pointAt(depth_est(n.row,
                    //                    n.col));

                    ceres::ResidualBlockId block_id{problem.AddResidualBlock(
                        FunctorNormalDistance::create(params_.ks, support_this, direction_this,
                                                      support_nn, direction_nn),
                        new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                              ceres::TAKE_OWNERSHIP),
                        &depth_est(row, col), &depth_est(n.row, n.col),
                        cloud.at(col, row).normal.data())};
                    normal_dist_ids.emplace_back(block_id);
                }

                problem.SetParameterization(cloud.at(col, row).normal.data(),
                                            new util_ceres::ConstantLengthParameterization<3>);
            }
        }
        ceres::Problem::EvaluateOptions options;
        options.residual_blocks = normal_dist_ids;
        double total_cost = 0.0;
        std::vector<double> residuals;
        problem.Evaluate(options, &total_cost, &residuals, nullptr, nullptr);
        int i{0};
        for (const auto& c : residuals) {
            LOG(INFO) << "residual cost at " << i << " is :" << c;
            i++;
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

    params_.solver.max_num_iterations = params_.max_iterations;
    params_.solver.max_solver_time_in_seconds = 120;
    ceres::Solve(params_.solver, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    LOG(INFO) << "Write output data";
    out.transform = util_ceres::fromQuaternionTranslation(rotation, translation);
    cv::eigen2cv(depth_est, out.image);
    out.cloud->reserve(rows * cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            //            LOG(INFO) << "Estimated depth for (" << col << "," << row
            //                      << "): " << depth_est(row, col);
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(col, row), support, direction);
            PointT p;
            p.getVector3fMap() = (support + direction * depth_est(row, col)).cast<float>();
            p.getNormalVector3fMap() = cloud.at(col, row).normal.cast<float>();
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
