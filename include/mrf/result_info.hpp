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
            << "number_of_image_points";
        return oss.str();
    }

    inline friend std::ostream& operator<<(std::ostream& os, const ResultInfo& o) {
        return os << o.optimization_successful << del << o.number_of_3d_points << del
                  << o.number_of_image_points;
    }

    bool optimization_successful{false};
    size_t number_of_3d_points{0};
    size_t number_of_image_points{0};

    bool has_covariance_depth{false};
    Eigen::MatrixXd covariance_depth;

    bool has_smoothness_costs{false};
    Eigen::MatrixXd smoothness_costs;

    bool has_normal_distance_costs{false};
    Eigen::MatrixXd normal_distance_costs;

    Eigen::Matrix<double, 7, 7> covariance_transform;
};
}
