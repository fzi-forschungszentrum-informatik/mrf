#pragma once

#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>

#include "data.hpp"
#include "export.hpp"

TEST(Data, Instantiation) {
    using namespace mrf;
    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT());
    const DataT::Ptr d{DataT::create(cl, cv::Mat::zeros(2, 2, CV_32FC1), Eigen::Affine3d::Identity())};
    std::cout << "\nTest data:\n" << *d;

    boost::filesystem::path path_name{"/tmp/test/data/"};
	boost::filesystem::create_directories(path_name);
	exportData(*d, path_name.string());
}
