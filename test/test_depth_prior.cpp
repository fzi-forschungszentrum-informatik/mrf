#include <io.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "depth_prior.hpp"
#include "solver.hpp"

TEST(DepthPrior, initialisation) {
    using namespace mrf;

    google::InitGoogleLogging("DepthPrior");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 50;
    constexpr size_t cols = 100;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT(20, 40, 10));
    cl->push_back(PointT(30, 40, 10));
    cl->push_back(PointT(25, 45, 10));
    const DataT in(cl, cv::Mat::eye(rows, cols, CV_32FC1), DataT::Transform::Identity());
    Parameters params;
    params.initialization = Parameters::Initialization::triangles;
    params.ks = 0;
    params.kd = 0;
    LOG(INFO) << "Params: " << params;
    Solver solver{cam, params};

    Data<pcl::PointXYZINormal> out;
    solver.solve(in, out);

    boost::filesystem::path path_name{"/tmp/test/depthPrior/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
}
