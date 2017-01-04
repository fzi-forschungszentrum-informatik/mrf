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

    Eigen::MatrixXd covariance_depth;
    Eigen::MatrixXd smoothness_costs;
    Eigen::MatrixXd normal_distance_costs;
    Eigen::Matrix<double, 7, 7> covariance_transform;
};
}
