#pragma once

#include <ostream>
#include <sstream>

namespace mrf {

struct ResultInfo {

    static constexpr char del = ','; ///< Delimiter

    /**
     * Get the parameter names and their order as string
     * @return
     */
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

    double t_prior{0};
    double t_solver{0};
    bool optimization_successful{false};
    size_t number_of_3d_points{0};
    size_t number_of_image_points{0};
    size_t iterations_used{0};

    bool has_covariance_depth{false};
    Eigen::MatrixXd covariance_depth;

    bool has_smoothness_costs{false};
    Eigen::MatrixXd smoothness_costs;

    bool has_normal_distance_costs{false};
    Eigen::MatrixXd normal_distance_costs;

    bool has_weights{false};
    Eigen::MatrixXd weights;

    float out_depth_min{0};
    float out_depth_max{0};

    Eigen::Matrix<double, 7, 7> covariance_transform;
};
}
