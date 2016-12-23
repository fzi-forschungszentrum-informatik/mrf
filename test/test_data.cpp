#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <io.hpp>
#include <pcl/point_types.h>

#include "data.hpp"

TEST(Data, Instantiation) {

    google::InitGoogleLogging("Data");
    google::InstallFailureSignalHandler();

    using namespace mrf;
    using PointT = pcl::PointXYZ;
    using DataT = Data<PointT>;
    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT());
    const DataT::Ptr d{
        DataT::create(cl, cv::Mat::zeros(2, 2, CV_32FC1))};
    LOG(INFO) << "Data: \n" << *d;
    boost::filesystem::path path_name{"/tmp/test/data/"};
    boost::filesystem::create_directories(path_name);
    exportData(*d, path_name.string());
}
