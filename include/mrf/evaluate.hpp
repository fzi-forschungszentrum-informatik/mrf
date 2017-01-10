#pragma once

#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include "data.hpp"
#include "quality.hpp"

namespace mrf {

inline double median(const cv::Mat& in) {
    std::vector<double> v;
    in.reshape(0, 1).copyTo(v);
    auto median = begin(v);
    std::advance(median, v.size() / 2);
    std::nth_element(begin(v), median, end(v));
    return *median;
}

inline double squaredSum(const cv::Mat& in) {
    cv::Mat sq;
    cv::pow(in, 2, sq);
    return cv::sum(sq)[0];
}

template <typename T, typename U>
Quality evaluate(const Data<T>& ref, const Data<U>& est) {

    Quality q;

    CHECK_EQ(est.image.size, ref.image.size) << "Images do not have the same size";
    const size_t size = est.image.rows * est.image.cols;
    q.depth_error = est.image - ref.image;

    using namespace cv;
    const Mat depth_error_abs{abs(q.depth_error)};
    q.depth_error_mean = sum(q.depth_error)[0] / size;
    q.depth_error_mean_abs = sum(depth_error_abs)[0] / size;
    q.depth_error_median = median(q.depth_error);
    q.depth_error_median_abs = median(depth_error_abs);
    q.depth_error_rms = std::sqrt(squaredSum(q.depth_error)) / size;

    CHECK_EQ(est.cloud->height, ref.cloud->height) << "Clouds do not have the same height";
    CHECK_EQ(est.cloud->width, ref.cloud->width) << "Clouds do not have the same width";
    size_t invalid_points{0};
    for (size_t r = 0; r < est.cloud->height; r++) {
        for (size_t c = 0; c < est.cloud->width; c++) {
            const T& p_ref{ref.cloud->at(c, r)};
            if (!std::isfinite(p_ref.normal_x) || !std::isfinite(p_ref.normal_y) ||
                !std::isfinite(p_ref.normal_z)) {
                invalid_points++;
                continue;
            }
            const Eigen::Vector3d n_est{
                est.cloud->at(c, r).getNormalVector3fMap().template cast<double>()};
            const Eigen::Vector3d n_ref{p_ref.getNormalVector3fMap().template cast<double>()};
            const Eigen::Vector3d delta{n_est - n_ref};
            const double dot_product{n_est.dot(n_ref)};
            q.normal_error_mean = q.normal_error_mean + delta;
            q.normal_error_mean_abs += delta.array().abs().matrix();
            q.normal_dot_product_mean += dot_product;
            q.normal_dot_product_mean_abs += std::abs(dot_product);
        }
    }
    const size_t normalizer{est.cloud->size() - invalid_points};
    q.normal_error_mean /= normalizer;
    q.normal_error_mean_abs /= normalizer;
    q.normal_dot_product_mean /= normalizer;
    q.normal_dot_product_mean_abs /= normalizer;

    return q;
}

inline size_t badMatchedPixels(const cv::Mat& in, const double& max_val) {
    size_t count{0};
    for (int r = 0; r < in.rows; r++) {
        for (int c = 0; c < in.cols; c++) {
            if (in.at<float>(r, c) > max_val) {
                count++;
            }
        }
    }
    return count;
}
}
