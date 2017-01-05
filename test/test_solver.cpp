#include <io.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "solver.hpp"

mrf::Data<pcl::PointXYZINormal> create(const size_t& rows, const size_t& cols) {
    using namespace mrf;

    //    Eigen::Affine3d tf{
    //        Eigen::AngleAxisd(10 * M_PI / 180, Eigen::Vector3d(1, 0.2, -4).normalized())};
    //    tf.translation() = Eigen::Vector3d(5, 0, -8);
    Eigen::Affine3d tf{Eigen::Affine3d::Identity()};

    using T = pcl::PointXYZINormal;
    const typename Data<T>::Cloud::Ptr cl{new typename Data<T>::Cloud};
    T p;
    p.getVector3fMap() = Eigen::Vector3f(cols - 1, 1, 0);
    p.getNormalVector3fMap() = Eigen::Vector3f(0, 0, -1);
    cl->push_back(p);
    p.getVector3fMap() = Eigen::Vector3f(1, rows - 1, 1);
    p.getNormalVector3fMap() = Eigen::Vector3f(0, 0, -1);
    cl->push_back(p);
    pcl::transformPointCloudWithNormals(*cl, *cl, tf);
    return Data<T>(cl, cv::Mat::eye(rows, cols, cv::DataType<uint8_t>::type), tf.inverse());
}

TEST(Solver, Solve) {
    using namespace mrf;

    google::InitGoogleLogging("Solver");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 20;
    constexpr size_t cols = 30;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    const Data<pcl::PointXYZINormal> in{create(rows, cols)};
    Data<pcl::PointXYZINormal> out;

    Parameters p("parameters.yaml");
    p.estimate_normals = false;
    p.pin_distances = true;
    p.pin_normals = true;
    Solver solver{cam, p};
    solver.solve(in, out);
    const Data<pcl::PointXYZINormal> debug {solver.getDebugInfo()};

    boost::filesystem::path path_name{"/tmp/test/solver/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
    exportData(debug, path_name.string() + "debug_");

    ASSERT_NEAR(pcl::transformPoint(out.cloud->at(cols - 1, 1), out.transform.cast<float>()).z, 0,
                1e-6);
    ASSERT_NEAR(pcl::transformPoint(out.cloud->at(1, rows - 1), out.transform.cast<float>()).z, 1,
                1e-6);
    ASSERT_NEAR(
        pcl::transformPoint(out.cloud->at(cols - 1, rows / 2), out.transform.cast<float>()).z, 0,
        1e-3);
}
