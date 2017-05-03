#pragma once

#include <ostream>
#include <sstream>
#include <Eigen/Geometry>

namespace mrf {
/** @brief Stores information about an optimization.
 *  Will be returned by the solver to the according tool requesting the optimization. */
struct ResultInfo {

    static constexpr char del = ','; ///< Delimiter

    /** @brief Get the parameter names and their order as string
     *  @return Result raw values seperated by the delimiter */
    inline static std::string header() {
        std::ostringstream oss;
        oss << "optimization_successful" << del << "number_of_3d_points" << del
            << "number_of_image_points" << del << "time_diff_prior" << del << "time_diff_solver"
            << del << "iterations_used" << del << "depth_max" << del << "depth_min";
        return oss.str();
    }

    inline friend std::ostream& operator<<(std::ostream& os, const ResultInfo& o) {
        return os << o.optimization_successful << del << o.number_of_3d_points << del
                  << o.number_of_image_points << del << o.t_prior << del << o.t_solver << del
                  << o.iterations_used << del << o.out_depth_max << del << o.out_depth_min;
    }

    double t_prior{0};                   ///< Time in seconds needed for the preprocessing
    double t_solver{0};                  ///< Time in seconds needed to solve the optimization
    bool optimization_successful{false}; ///< Success of the optimization
    size_t number_of_3d_points{
        0}; ///< Number of the laserscanner measurements in front of the image
    size_t number_of_image_points{0}; ///< Number of pixel in the camera image
    size_t iterations_used{0};        ///< Number of iterations the solver used

    bool has_covariance_depth{false}; ///< Are covariances saved
    Eigen::MatrixXd covariance_depth; ///< Calculated covariances

    bool has_smoothness_costs{false}; ///<
    Eigen::MatrixXd smoothness_costs; ///<

    bool has_normal_distance_costs{false}; ///<
    Eigen::MatrixXd normal_distance_costs; ///<

    bool has_weights{false}; ///< Are smoothness weights saved
    Eigen::MatrixXd weights; ///< Smoothness weights

    float out_depth_min{0}; ///< Minimum estimated depth
    float out_depth_max{0}; ///< Maximum estimated depth

    Eigen::Matrix<double, 7, 7> covariance_transform; ///<
};
}
