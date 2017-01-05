#pragma once

#include <ostream>
#include <sstream>
#include <opencv2/core/core.hpp>

namespace mrf {

struct Quality {

    static constexpr char del = ','; ///< Delimiter

    /**
     * Get the parameter names and their order as string
     * @return
     */
    inline static std::string header() {
        std::ostringstream oss;
        oss << "depth_error_mean" << del << "depth_error_mean_abs" << del << "depth_error_median"
            << del << "depth_error_median_abs" << del << "depth_error_rms";
        return oss.str();
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Quality& o) {
        return os << o.depth_error_mean << del << o.depth_error_mean_abs << del
                  << o.depth_error_median << del << o.depth_error_median_abs << del
                  << o.depth_error_rms;
    }

    cv::Mat depth_error;
    double depth_error_mean{0};
    double depth_error_mean_abs{0};
    double depth_error_median{0};
    double depth_error_median_abs{0};
    double depth_error_rms{0};
};
}
