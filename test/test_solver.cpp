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

    constexpr size_t rows = 10;
    constexpr size_t cols = 10;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    DataT::Image img{cv::Mat::eye(rows, cols, CV_32FC1)};
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    PointT p;
    p.x = 1;
    p.y = rows - 1;
    p.z = 1;
    cl->push_back(p);
    p.x = cols - 1;
    p.y = 1;
    p.z = 0;
    cl->push_back(p);
    DataT d(cl, img, DataT::Transform::Identity());

    Solver solver{cam};
    solver.solve(d);

    boost::filesystem::path path_name{"/tmp/test/solver/"};
    boost::filesystem::create_directories(path_name);
    exportData(d, path_name.string());
    exportDepthImage<PointT>(d, cam, path_name.string());
}