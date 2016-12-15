#pragma once

#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "data.hpp"

TEST(Data, Instantiation) {
    using namespace mrf;
    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    DataT::Image img(2, 2, CV_32FC1, 0);
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT());
    const DataT::Transform tf{DataT::Transform::Identity()};
    const DataT::Ptr d{DataT::create(cl, img, tf)};
    std::cout << "\nTest data:\n" << *d;
}
