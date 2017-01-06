#include <io.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "prior.hpp"
#include "solver.hpp"

TEST(DepthPrior, initialisation) {
    using namespace mrf;

    google::InitGoogleLogging("Prior");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 10;
    constexpr size_t cols = 20;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT(1, 1, 10));
    cl->push_back(PointT(1, cols - 2, 10));
    cl->push_back(PointT(rows - 2, 1, 10));
    const cv::Mat img{cv::Mat::zeros(rows, cols, CV_32FC3)};
    const DataT in(cl, img, DataT::Transform::Identity());
    Parameters params("parameters.yaml");
    params.initialization = Parameters::Initialization::triangles;
    params.ks = 0;
    params.kd = 0;
    LOG(INFO) << "Params: " << params;
    Solver solver{cam, params};

    Data<pcl::PointXYZINormal> out;
    solver.solve(in, out);

    boost::filesystem::path path_name{"/tmp/test/depth_prior/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
}
