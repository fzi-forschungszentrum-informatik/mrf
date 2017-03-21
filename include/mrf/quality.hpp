#pragma once

#include <ostream>
#include <sstream>
#include <opencv2/core/core.hpp>

namespace mrf {

struct Quality {

    static constexpr char del = ','; ///< Delimiter

    /** @brief Get the parameter names and their order as string
     *  @return Quality raw values seperated by a delimiter */
    inline static std::string header() {
        std::ostringstream oss;
        // clang-format off
        oss << "depth_error_mean" << del
        		<< "depth_error_mean_abs" << del
				<< "depth_error_median" << del
				<< "depth_error_median_abs" << del
				<< "depth_error_rms" << del
				<< "normal_error_x_mean" << del
				<< "normal_error_y_mean" << del
				<< "normal_error_z_mean" << del
				<< "normal_error_x_mean_abs" << del
				<< "normal_error_y_mean_abs" << del
				<< "normal_error_z_mean_abs" << del
				<< "normal_dot_product_mean" << del
				<< "normal_dot_product_mean_abs" << del
				<< "ref_distances_evaluated" << del
				<< "ref_normals_evaluated";
        // clang-format on
        return oss.str();
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Quality& o) {
        // clang-format off
        return os << o.depth_error_mean << del
        		<< o.depth_error_mean_abs << del
                << o.depth_error_median << del
				<< o.depth_error_median_abs << del
                << o.depth_error_rms << del
                << o.normal_error_mean.x() << del
                << o.normal_error_mean.y() << del
                << o.normal_error_mean.z() << del
                << o.normal_error_mean_abs.x() << del
                << o.normal_error_mean_abs.y() << del
                << o.normal_error_mean_abs.z() << del
                << o.normal_dot_product_mean << del
        		<< o.normal_dot_product_mean_abs << del
				<< o.ref_distances_evaluated << del
				<< o.ref_normals_evaluated;
        // clang-format on;
    }

    cv::Mat depth_error;                ///< Image with the depth error
    double depth_error_mean{0};         ///< Mean depth error of all points
    double depth_error_mean_abs{0};     ///< Mean absolute depth error of all points
    double depth_error_median{0};       ///< Median depth error of all points
    double depth_error_median_abs{0};   ///< Median absolute depth error of all points
    double depth_error_rms{0};          ///< Root mean-square error of all points

    Eigen::Vector3d normal_error_mean{Eigen::Vector3d::Zero()};     ///<
    Eigen::Vector3d normal_error_mean_abs{Eigen::Vector3d::Zero()}; ///<
    double normal_dot_product_mean{0};          ///< Mean of the dot product of all estimated and calculated normals
    double normal_dot_product_mean_abs{0};      ///< Absolute mean of the dot product of all estimated and calculated normals

    size_t ref_distances_evaluated{0};          ///< Number of points evaluated
    size_t ref_normals_evaluated{0};            ///< Numebr of normals evaluated
};
}
