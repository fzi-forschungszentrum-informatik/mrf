#include <io.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>

#include "camera_model_ortho.h"
#include "downsample.hpp"
#include "noise.hpp"
#include "solver.hpp"

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
            : equidistant{false}, rows_inbetween{20}, cols_inbetween{10}, seedpoint_number{100},
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

template <typename T>
mrf::Data<T> createDense(const size_t& rows, const size_t& cols) {
    using namespace mrf;
    cv::Mat img{cv::Mat::zeros(rows, cols, CV_32FC1)};
    const typename Data<T>::Cloud::Ptr cl{new typename Data<T>::Cloud};
    cl->resize(rows * cols);

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < 0.3 * cols; col++) {
            img.at<float>(row, col) = 1;
            cl->points[row * cols + col] = pcl::PointXYZ(row, col, img.at<float>(row, col));
        }
    }
    for (size_t row = 0; row < 0.7 * rows; row++) {
        for (size_t col = 0.3 * cols; col < cols; col++) {
            img.at<float>(row, col) = row / (0.7 * rows);
            cl->points[row * cols + col] = pcl::PointXYZ(row, col, img.at<float>(row, col));
        }
    }
    for (size_t row = 0.7 * rows; row < rows; row++) {
        for (size_t col = 0.3 * cols; col < cols; col++) {
            img.at<float>(row, col) = 0;
            cl->points[row * cols + col] = pcl::PointXYZ(row, col, img.at<float>(row, col));
        }
    }
    cl->width = cols;
    cl->height = rows;
    return Data<T>(cl, img, Eigen::Affine3d::Identity());
}

TEST(Groundtruth, loadGT) {
    using namespace mrf;
    google::InitGoogleLogging("Groundtruth");
    google::InstallFailureSignalHandler();
    using PointT = pcl::PointXYZ;

    LOG(INFO) << "Set Parameters";
    constexpr size_t cols = 1000;
    constexpr size_t rows = 500;
    GroundTruthParams params;
    params.equidistant = true;
    LOG(INFO) << "Load Groundtruth Data";
    const Data<PointT> gt_data{createDense<PointT>(rows, cols)};
    LOG(INFO) << "dense cloud size: " << gt_data.cloud->size();

    LOG(INFO) << "Load Sparse Data";
    typename Data<PointT>::Cloud::Ptr sparse{new typename Data<PointT>::Cloud};
    if (params.equidistant) {
        LOG(INFO) << "Equidistant";
        sparse = downsampleEquidistant<PointT>(gt_data.cloud, params.rows_inbetween,
                                               params.cols_inbetween);
    } else {
        LOG(INFO) << "Random";
        sparse = downsampleRandom<PointT>(gt_data.cloud, params.seedpoint_number);
    }
    if (params.addCloudNoise) {
        LOG(INFO) << "Add Noise";
        sparse = addNoise<PointT>(gt_data.cloud, params.noise_sigma, params.noise_sigma,
                                  params.noise_sigma);
    }

    LOG(INFO) << "Solve";
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};
    Data<PointT> in(sparse, gt_data.image, gt_data.transform);
    Data<pcl::PointXYZINormal> out;
    Solver solver{cam, Parameters("parameters.yaml")};
    solver.solve(in, out);

    LOG(INFO) << "Write to file";
    boost::filesystem::path path_name{"/tmp/test/gt/solver/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
    exportDepthImage<PointT>(in, cam, path_name.string());
    exportGradientImage(in.image, path_name.string());
}
