#pragma once

#include <fstream>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace mrf{


float absoluteDifference(const Eigen::MatrixXd& gt_data, const Eigen::MatrixXd& result_data) {
    if (gt_data.size() != result_data.size()) {
        return -1;
    }
    return (gt_data - result_data).array().abs().sum();
}

float meanSquareError(const Eigen::MatrixXd& gt_data, const Eigen::MatrixXd& result_data) {
    if (gt_data.size() != result_data.size()) {
        return -1;
    }
    return ((gt_data - result_data) * (gt_data - result_data)).sum() / gt_data.size();
}

float rootMeanSquareError(const Eigen::MatrixXd& gt_data, const Eigen::MatrixXd& result_data) {
    if (gt_data.size() != result_data.size()) {
        return -1;
    }
    return sqrt(meanSquareError(gt_data, result_data));
}

float badMatchedPixels(const Eigen::MatrixXd& gt_data, const Eigen::MatrixXd& result_data,
                       const float delta_max) {
    if (gt_data.size() != result_data.size()) {
        return -1;
    }
    int count = 0;
    for (int i = 0; i < gt_data.size(); i++) {
        if (std::abs(gt_data(i) - result_data(i)) > delta_max) {
            count++;
        }
    }
    return count / gt_data.size();
}

cv::Mat heatmap(Eigen::MatrixXd& depth_gt, Eigen::MatrixXd& depth_est){
	cv::Mat out;

	return out;
}


void evaluate_old(Eigen::MatrixXd& depth_gt, Eigen::MatrixXd& depth_est, std::stringstream& stream) {
    stream << "Evaluation: " << std::endl;
    stream << "absolute_difference: " << absoluteDifference(depth_gt, depth_est) << std::endl;
    stream << "mean_square_error: " << meanSquareError(depth_gt, depth_est) << std::endl;
    stream << "root_mean_square_error: " << rootMeanSquareError(depth_gt, depth_est) << std::endl;
    stream << "bad_matched_pixels: " << badMatchedPixels(depth_gt, depth_est, 25)
           << std::endl;
    stream << std::endl;
}

}
