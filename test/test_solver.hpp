#pragma once

#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "solver.hpp"

TEST(Solver, Instantiation) {
    using namespace mrf;

    google::InitGoogleLogging("TestSolver");

    constexpr size_t width = 10;
    constexpr size_t height = 10;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(width, height)};

    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    DataT::Image img{cv::Mat::zeros(width, height, CV_32FC1)};
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    PointT p;
    p.z = 1;
    cl->push_back(p);
    const DataT::Transform tf{DataT::Transform::Identity()};
    DataT d(cl, img, tf);

    LOG(INFO) << "Construct";
    Solver solver{cam};
    LOG(INFO) << "Solve";
    solver.solve(d);
}
