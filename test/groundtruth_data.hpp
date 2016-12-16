#pragma once

#include <fstream>
#include <Eigen/Eigen>
#include <opencv/cxcore.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <ros/package.h>

namespace mrf {
using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;

struct GroundTruthParams {
    bool equidistant;
    int rows_inbetween;
    int cols_inbetween;
    int seedpoint_number;

    bool addCloudNoise;
    bool addImageNoise;
    bool addImageBlur;
    int blur_size;
    double noise_sigma;

    GroundTruthParams()
            : equidistant{true}, rows_inbetween{20}, cols_inbetween{10}, seedpoint_number{100},
              addCloudNoise{false}, addImageNoise{false}, addImageBlur{false}, blur_size{3},
              noise_sigma{2} {};

    friend std::ostream& operator<<(std::ostream& os, const GroundTruthParams& f) {
        os << "Groundtruth Parameters: " << std::endl;
        os << "equidistant: " << f.equidistant << std::endl;
        os << "rows_inbetween: " << f.rows_inbetween << std::endl;
        os << "cols_inbetween: " << f.cols_inbetween << std::endl;
        os << "seedpoint_number: " << f.seedpoint_number << std::endl;
        os << "addCloudNoise: " << f.addCloudNoise << std::endl;
        os << "addImageNoise: " << f.addImageNoise << std::endl;
        os << "addImageBlur: " << f.addImageBlur << std::endl;
        os << "blur_size: " << f.blur_size << std::endl;
        os << "noise_sigma: " << f.noise_sigma << std::endl;
        os << std::endl;
    }
};


void loadCameraImage(cv::Mat& image_out) {
    //const std::string path{ros::package::getPath("mrf")};
    std::stringstream image_ss;
    image_ss << "/home/roxin/Desktop/U/catkin_ws/src/mrf/test/res/groundtruth/image.png";// << "/test/res/groundtruth/image.png";
    std::cout<< "path: "<< image_ss.str();
    image_out = cv::imread(image_ss.str());
}

void loadDenseCloud(const Cloud::Ptr& cloud_dense) {
    const std::string path{ros::package::getPath("mrf")};
    std::stringstream image_ss;
    image_ss << path << "/test/res/groundtruth/gt_dense.pcd";
    //pcl::io::loadPCDFile<Point>(image_ss.str(), *cloud_dense);
}

void addNoiseToCloud(const Cloud::Ptr& cloud_sparse, const GroundTruthParams params) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0, params.noise_sigma);

    for (int i = 0; i < cloud_sparse->points.size(); i++) {
        cloud_sparse->points[i].x += distribution(generator);
        cloud_sparse->points[i].y += distribution(generator);
        cloud_sparse->points[i].z += distribution(generator);
    }
}

void loadCloudSparse(const Cloud::Ptr& cloud_dense, const Cloud::Ptr& cloud_sparse, const int width,
                     const int height, const GroundTruthParams params) {
    cloud_sparse->points.reserve(width * height);

    if (params.equidistant) { //> equidistant randomiser
        for (int i = 1; i < height; i += params.rows_inbetween) {
            for (int j = 1; j < width; j += params.cols_inbetween) {
                cloud_sparse->points.emplace_back(cloud_dense->points.at(i * width + j));
            }
        }

    } else {

        std::srand(std::time(0));
        for (int i = 0; i < params.seedpoint_number; i++) {
            cloud_sparse->points.emplace_back(
                cloud_dense->points[(std::rand() % (int)(width * height + 1))]);
        }
    }
    cloud_sparse->width = 1;
    cloud_sparse->height = cloud_sparse->points.size();

    if (params.addCloudNoise) {
        addNoiseToCloud(cloud_sparse, params);
    }
}
} // end of namespace
