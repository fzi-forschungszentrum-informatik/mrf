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
        oss << "depth_error_avg";
        return oss.str();
    }

    inline friend std::ostream& operator<<(std::ostream& os, const ResultInfo& o) {
        return os << o.depth_error_avg;
    }

    cv::Mat depth_error;
    double depth_error_avg{0};
};
}
