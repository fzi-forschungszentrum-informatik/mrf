#include <io.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "solver.hpp"

template <typename T>
mrf::Data<T> create(const size_t& rows, const size_t& cols) {
    using namespace mrf;

    Eigen::Affine3d tf{
        Eigen::AngleAxisd(10 * M_PI / 180, Eigen::Vector3d(1, 0.2, -4).normalized())};
    tf.translation() = Eigen::Vector3d(5, 0, -8);

    const typename Data<T>::Cloud::Ptr cl{new typename Data<T>::Cloud};
    cl->push_back(pcl::transformPoint(T(cols - 1, 1, 0), tf.cast<float>()));
    cl->push_back(pcl::transformPoint(T(1, rows - 1, 1), tf.cast<float>()));
    return Data<T>(cl, cv::Mat::eye(rows, cols, CV_32FC1), tf.inverse());
}

TEST(Solver, SolveSkew) {
    using namespace mrf;
    using PointT = pcl::PointXYZINormal;
    google::InitGoogleLogging("SolverSkew");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 20;
    constexpr size_t cols = 30;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};
    const typename Data<PointT>::Cloud::Ptr cl{new typename Data<PointT>::Cloud};
    PointT p;
    p.x = 1;
    p.y = rows - 1;
    p.z = 10;
    p.normal_x = 0;
    p.normal_y = 0;
    p.normal_z = 1;
    cl->push_back(p);
    p.x = cols - 1;
    p.y = 1;
    p.z = 2;
    p.normal_x = 0;
    p.normal_y = 0;
    p.normal_z = 1;
    cl->push_back(p);
    const Data<PointT> in{cl,cv::Mat::eye(rows, cols, CV_32FC1),Eigen::Affine3d::Identity()};
    Data<PointT> out;

    Solver solver{cam, Parameters("parameters.yaml")};
    solver.solve(in, out);

    boost::filesystem::path path_name{"/tmp/test/solver_skew/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
    exportGradientImage(in.image, path_name.string());
}

TEST(Solver, Solve) {
    using namespace mrf;

    google::InitGoogleLogging("Solver");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 20;
    constexpr size_t cols = 30;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    const Data<pcl::PointXYZ> in{create<pcl::PointXYZ>(rows, cols)};
    Data<pcl::PointXYZINormal> out;

    Solver solver{cam, Parameters("parameters.yaml")};
    solver.solve(in, out);

    boost::filesystem::path path_name{"/tmp/test/solver/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
    exportGradientImage(in.image, path_name.string());

    ASSERT_NEAR(pcl::transformPoint(out.cloud->at(cols - 1, 1), out.transform.cast<float>()).z, 0,
                1e-4);
    ASSERT_NEAR(pcl::transformPoint(out.cloud->at(1, rows - 1), out.transform.cast<float>()).z, 1,
                1e-4);
    ASSERT_NEAR(
        pcl::transformPoint(out.cloud->at(cols - 1, rows / 2), out.transform.cast<float>()).z, 0,
        1e-4);
}
