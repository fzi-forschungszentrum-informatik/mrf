#pragma once

#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

inline double smoothnessWeight(const Pixel& p, const Pixel& neighbor, const double& threshold,
                               const double& alpha, const double& weight_min) {
    const double abs_diff{std::abs(p.val - neighbor.val)};
    if (abs_diff < threshold) {
        return 1;
    } else {
        return std::max(exp(-alpha * std::fabs(abs_diff - threshold)), weight_min);
    }
}
}
