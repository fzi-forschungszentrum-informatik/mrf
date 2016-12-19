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
    p.x = 1;
    p.y = 1;
    p.z = 10;
    cl->push_back(p);
    DataT d(cl, img, DataT::Transform::Identity());

    LOG(INFO) << "Construct";
    Solver solver{cam};
    LOG(INFO) << "Solve";
    solver.solve(d);
}
