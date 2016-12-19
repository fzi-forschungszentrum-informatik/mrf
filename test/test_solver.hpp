#pragma once

#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>

#include "camera_model_ortho.h"
#include "solver.hpp"

TEST(Solver, Instantiation) {
    using namespace mrf;

    google::InitGoogleLogging("TestSolver");

    constexpr size_t rows = 200;
    constexpr size_t cols = 200;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(rows, cols)};

    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    DataT::Image img{cv::Mat::eye(rows, cols, CV_32FC1)};
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    PointT p;
    p.x = 1;
    p.y = cols - 1;
    p.z = 10;
    p.x = rows - 1;
    p.y = 1;
    p.z = 0;
    cl->push_back(p);
    DataT d(cl, img, DataT::Transform::Identity());

    Solver solver{cam};
    solver.solve(d);

    /**
     * Write data to files
     * 1. Raw image
     * 2. Estimated depth image
     * 3. Point cloud
     */
    boost::filesystem::path path_name{"/tmp/test/solver/"};
    boost::filesystem::create_directories(path_name);
    std::string file_name;

    file_name = path_name.string() + "raw.png";
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    cv::imwrite(file_name, img);

    cv::Mat img_depth{cv::Mat::zeros(rows, cols, CV_32FC1)};
    const Eigen::Matrix3Xd pts_3d{d.transform *
                                  d.cloud->getMatrixXfMap().topRows<3>().cast<double>()};
    const double depth_max{pts_3d.colwise().norm().maxCoeff()};
    const double depth_min{pts_3d.colwise().norm().minCoeff()};
    const double scale{1 / (depth_max - depth_min)};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, pts_3d.cols())};
    std::vector<bool> in_img{cam->getImagePoints(pts_3d, img_pts_raw)};
    for (size_t c = 0; c < in_img.size(); c++) {
        const int row = img_pts_raw(0, c);
        const int col = img_pts_raw(1, c);
        img_depth.at<float>(col, row) = scale * (pts_3d.col(c).norm() - depth_min);
    }
    cv::normalize(img_depth, img_depth, 0, 255, cv::NORM_MINMAX);

    file_name = path_name.string() + "depth_est.png";
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::imwrite(file_name, img_depth);

    file_name = path_name.string() + "depth_est.pcd";
    LOG(INFO) << "Writing cloud to '" << file_name << "'.";
    pcl::io::savePCDFile<PointT>(file_name, *(d.cloud), true);
}
