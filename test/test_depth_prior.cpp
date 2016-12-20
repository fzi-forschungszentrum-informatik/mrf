
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "depth_prior.hpp"
#include "export.hpp"
#include "solver.hpp"

TEST(DepthPrior, Mean) {
    using namespace mrf;

    google::InitGoogleLogging("DepthPrior");
    google::InstallFailureSignalHandler();
    LOG(INFO) << "Depth prior Testing";

    constexpr size_t rows = 15;
    constexpr size_t cols = 15;
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
    p.z = 10;
    cl->push_back(p);
    p.x = 2;
    p.y = rows - 1;
    p.z = 1;
    cl->push_back(p);
    p.x = 1;
    p.y = rows - 1;
    p.z = 1;
    cl->push_back(p);
    p.x = 1;
    p.y = rows - 1;
    p.z = 1;
    cl->push_back(p);
    p.x = 1;
    p.y = rows - 1;
    p.z = 1;
    cl->push_back(p);
    p.x = 1;
    p.y = rows - 1;
    p.z = 1;

    DataT d(cl, img, DataT::Transform::Identity());
    Parameters params;
    params.initialization = Initialization::triangles;
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
