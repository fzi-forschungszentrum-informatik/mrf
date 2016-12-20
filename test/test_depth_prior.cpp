
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "depth_prior.hpp"
#include "export.hpp"
#include "solver.hpp"

TEST(DepthPrior, initialisation) {
    using namespace mrf;

    google::InitGoogleLogging("DepthPrior");
    google::InstallFailureSignalHandler();
    LOG(INFO) << "Depth prior Testing";

    constexpr size_t rows = 50;
    constexpr size_t cols = 100;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    DataT::Image img{cv::Mat::eye(rows, cols, CV_32FC1)};
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    PointT p;
    p.x = 20;
    p.y = 40;
    p.z = 10;
    cl->push_back(p);
    p.x = 30;
    p.y = 40;
    p.z = 10;
    cl->push_back(p);
    p.x = 25;
    p.y = 45;
    p.z = 10;
    cl->push_back(p);
    DataT d(cl, img, DataT::Transform::Identity());
    Parameters params;
    params.initialization = Parameters::Initialization::triangles;
    params.ks = 0;
    params.kd = 0;
    LOG(INFO) << "params set to";
    LOG(INFO) << params;
    Solver solver{cam, params};
    solver.solve(d);

    boost::filesystem::path path_name{"/tmp/test/depthPrior/"};
    boost::filesystem::create_directories(path_name);
    exportData(d, path_name.string());
    exportDepthImage<PointT>(d, cam, path_name.string());
}
