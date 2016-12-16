#pragma once

#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "camera_model_ortho.h"
#include "solver.hpp"

TEST(Solver, Instantiation) {
    using namespace mrf;

    constexpr size_t width = 100;
    constexpr size_t height = 100;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(width, height)};

    using PointT = pcl::PointXYZ;
	using DataT = Data<PointT>;
	DataT::Image img(width, height, CV_32FC1, 0);
	const DataT::Cloud::Ptr cl{new DataT::Cloud};
	PointT p;
	p.z = 1;
	cl->push_back(p);
	const DataT::Transform tf{DataT::Transform::Identity()};
	DataT d(cl, img, tf);

    Solver solver{cam};
    solver.solve(d);
}
