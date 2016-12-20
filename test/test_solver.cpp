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
    img = cv::Mat::eye(rows, cols, CV_32FC1);

    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT(1, rows - 1, 1));
    cl->push_back(PointT(cols - 1, 1, 0));
    const DataT in(cl, img, DataT::Transform::Identity());
    DataT out;

    Solver solver{cam, Parameters("parameters.yaml")};
    solver.solve(in, out);

    boost::filesystem::path path_name{"/tmp/test/solver/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
    exportGradientImage(in.image, path_name.string());

    ASSERT_NEAR(out.cloud->at(cols - 1, 1).z, 0, 1e-4);
    ASSERT_NEAR(out.cloud->at(1, rows - 1).z, 1, 1e-4);
    ASSERT_NEAR(out.cloud->at(cols - 1, rows / 2).z, 0, 1e-4);
}
