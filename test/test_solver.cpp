#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "export.hpp"
#include "solver.hpp"

TEST(Solver, Instantiation) {
    using namespace mrf;

    google::InitGoogleLogging("Solver");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 200;
    constexpr size_t cols = 300;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    cv::Mat img;
//    cv::hconcat(cv::Mat::zeros(rows, cols / 2, CV_32FC1), cv::Mat::ones(rows, cols / 2, CV_32FC1), img);
    img = cv::Mat::eye(rows, cols, CV_32FC1);

    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT(1, rows - 1, 1));
    cl->push_back(PointT(cols - 1, 1, 0));
//    cl->push_back(PointT(50, 40, 50));
//    cl->push_back(PointT(60, 50, 50));
//    cl->push_back(PointT(60, 40, 50));
    DataT d(cl, img, DataT::Transform::Identity());

    Solver solver{cam, Parameters("parameters.yaml")};
    solver.solve(d);

    boost::filesystem::path path_name{"/tmp/test/solver/"};
    boost::filesystem::create_directories(path_name);
    exportData(d, path_name.string());
    exportDepthImage<PointT>(d, cam, path_name.string());
    exportGradientImage(d.image, path_name.string());

    ASSERT_NEAR(d.cloud->at(cols - 1, 1).z, 0, 1e-4);
    ASSERT_NEAR(d.cloud->at(1, rows - 1).z, 1, 1e-4);
    ASSERT_NEAR(d.cloud->at(cols - 1, rows / 2).z, 0, 1e-4);
}
