#include <io.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>

#include "camera_model_ortho.h"
#include "solver.hpp"

using namespace mrf;

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;
using DataT = Data<PointT>;
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
    cloud_sparse->width = cloud_sparse->points.size();
    cloud_sparse->height = 1;

    if (params.addCloudNoise) {
        addNoiseToCloud(cloud_sparse, params);
    }
}

TEST(Groundtruth, loadGT) {
    google::InitGoogleLogging("Groundtruth");
    google::InstallFailureSignalHandler();

    /**
     * Create Data
     */
    constexpr size_t cols = 1384;
    constexpr size_t rows = 1032;
    cv::Mat img{cv::Mat::zeros(rows, cols, CV_32FC1)};
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < 0.3 * cols; col++) {
            img.at<float>(row, col) = 1;
        }
    }
    for (size_t row = 0; row < 0.7 * rows; row++) {
        for (size_t col = 0.3 * cols; col < cols; col++) {
            img.at<float>(row, col) = row / (0.7 * rows);
        }
    }
    for (size_t row = 0.7 * rows; row < rows; row++) {
        for (size_t col = 0.3 * cols; col < cols; col++) {
            img.at<float>(row, col) = 0;
        }
    }

    const Cloud::Ptr cloud_dense{new Cloud};
    const Cloud::Ptr cloud_sparse{new Cloud};

    pcl::io::loadPCDFile<PointT>("gt_dense.pcd", *cloud_dense);

    /**
     * Load GT Data
     */
    GroundTruthParams params_gt;
    LOG(INFO) << "Test GroundTruthParams: " << params_gt;

    loadCloudSparse(cloud_dense, cloud_sparse, cols, rows, params_gt);
    LOG(INFO) << "cloud_sparse points: " << cloud_sparse->points.size() << std::endl;
    const DataT gt_data(cloud_dense, img, DataT::Transform::Identity());

    const DataT::Cloud::Ptr cl{new DataT::Cloud};
    cl->push_back(PointT(1, 1, 500));
    cl->push_back(PointT(cols - 1, rows - 1, 50));
    cl->push_back(PointT(845, 354, 271));

    /**
    * Solver
    */
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};
    DataT in(cloud_sparse, img, DataT::Transform::Identity());
    DataT out;
    Solver solver{cam, Parameters("parameters.yaml")};
    solver.solve(in, out);

    /**
     * Write output data
     */
    boost::filesystem::path path_name{"/tmp/test/gt/solver/"};
    boost::filesystem::create_directories(path_name);
    exportData(in, path_name.string() + "in_");
    exportData(out, path_name.string() + "out_");
    exportDepthImage<PointT>(in, cam, path_name.string());
    exportGradientImage(in.image, path_name.string());
}
