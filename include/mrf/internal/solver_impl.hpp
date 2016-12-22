#include <limits>
#include <map>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <util_ceres/eigen_quaternion_parameterization.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

#include "../cloud_preprocessing.hpp"
#include "../depth_prior.hpp"
#include "../functor_distance.hpp"
#include "../functor_normal.hpp"
#include "../functor_normal_distance.hpp"
#include "../functor_normal_smoothness.hpp"
#include "../functor_smoothness.hpp"
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
            p.val = img.at<float>(p.row, p.col);
            LOG(INFO) << "Normal at:  point (" << p.row << ", " << p.col
                      << ") is : " << cl->at(c).getNormalVector3fMap().transpose();
            projection.insert(std::make_pair(p, cl->at(c)));
            projection_tf.insert(std::make_pair(p, cl_tf->at(c)));
        }
    }
    LOG(INFO) << "projection size is: " << projection.size();

    LOG(INFO) << "Create optimization problem";
    ceres::Problem problem(params_.problem);
    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd certainty{Eigen::MatrixXd::Zero(rows, cols)};
    getDepthEst(depth_est, certainty, projection_tf, camera_, params_.initialization,
                params_.neighbor_search);

    LOG(INFO) << "Create Normals Eigen Vectors";
    Eigen::MatrixXd normal_x_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd normal_y_est{Eigen::MatrixXd::Zero(rows, cols)};
    Eigen::MatrixXd normal_z_est{Eigen::MatrixXd::Zero(rows, cols)};
    for (auto const& el : projection) {
        LOG(INFO) << "el: normals: (" << el.second.normal_x << ", " << el.second.normal_y << ", "
                  << el.second.normal_z << ")";
        normal_x_est(el.first.row, el.first.col) = el.second.normal_x;
        normal_y_est(el.first.row, el.first.col) = el.second.normal_y;
        normal_z_est(el.first.row, el.first.col) = el.second.normal_z;
    }

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
        functors_distance.emplace_back(FunctorDistance::create(
            el.second.getVector3fMap().cast<double>(), params_.kd, support, direction));
        problem.AddResidualBlock(functors_distance.back()->toCeres(), params_.loss_function.get(),
                                 &depth_est(el.first.row, el.first.col), rotation.coeffs().data(),
                                 translation.data());
        /**
         * Add Normal Costs
         */
        functor_normal.emplace_back(
            FunctorNormal::create(el.second.getNormalVector3fMap().cast<double>(), params_.kd));
        problem.AddResidualBlock(
            functor_normal.back()->toCeres(), params_.loss_function.get(),
            &normal_x_est(el.first.row, el.first.col), &normal_y_est(el.first.row, el.first.col),
            &normal_z_est(el.first.row, el.first.col), rotation.coeffs().data());
    }

    LOG(INFO) << "Add Normal Smoothness Costs";
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            const Pixel p(col, row, img.at<float>(row, col));
            // LOG(INFO) << "px: "<< normal_x_est(row, col) << ", py: "<<normal_y_est(row, col)<<",
            // nz: "<< normal_z_est(row, col);
            const std::vector<Pixel> neighbors{getNeighbors(p, img, params_.neighborhood)};
            for (auto const& n : neighbors) {
                // LOG(INFO) << "nx: "<< normal_x_est(n.row, n.col) << ", ny: "<<normal_y_est(n.row,
                // n.col)<<", nz: "<< normal_z_est(n.row, n.col);
                problem.AddResidualBlock(
                    FunctorNormalSmoothness::create(smoothnessWeight(p, n, params_) * params_.ks),
                    new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                          ceres::TAKE_OWNERSHIP),
                    &normal_x_est(row, col), &normal_y_est(row, col), &normal_z_est(row, col),
                    &normal_x_est(n.row, n.col), &normal_y_est(n.row, n.col),
                    &normal_z_est(n.row, n.col));
            }
        }
    }

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
                // LOG(INFO) << "nx: "<< normal_x_est(n.row, n.col) << ", ny: "<<normal_y_est(n.row,
                // n.col)<<", nz: "<< normal_z_est(n.row, n.col);
                problem.AddResidualBlock(
                    FunctorNormalDistance::create(params_.ks, support_this, direction_this,
                                                  support_nn, direction_nn),
                    new ceres::ScaledLoss(new ceres::TrivialLoss, 1. / neighbors.size(),
                                          ceres::TAKE_OWNERSHIP),
                    &depth_est(row, col), &depth_est(n.row, n.col), &normal_x_est(row, col),
                    &normal_y_est(row, col), &normal_z_est(row, col));
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
    out.cloud->reserve(rows * cols);
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            //            LOG(INFO) << "Estimated depth for (" << col << "," << row
            //                      << "): " << depth_est(row, col);
            Eigen::Vector3d support, direction;
            camera_->getViewingRay(Eigen::Vector2d(col, row), support, direction);
            PointT p;
            p.getVector3fMap() = (support + direction * depth_est(row, col)).cast<float>();
            p.getNormalVector3fMap() =
                Eigen::Vector3f(normal_x_est(row, col), normal_y_est(row, col),
                                normal_z_est(row, col))
                    .cast<float>();
            out.cloud->push_back(
                pcl::transformPoint(p, out.transform.inverse().template cast<float>()));
        }
    }
    out.cloud->width = cols;
    out.cloud->height = rows;

    out.transform = util_ceres::fromQuaternionTranslation(rotation, translation);
    cv::eigen2cv(depth_est, out.image);

    LOG(INFO) << "Write info";
    ResultInfo info;
    info.optimization_successful = summary.IsSolutionUsable();
    info.number_of_3d_points = projection.size();
    info.number_of_image_points = rows * cols;
    return info;
}
}
