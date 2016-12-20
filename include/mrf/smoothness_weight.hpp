#pragma once

#include "pixel.hpp"
#include "parameters.hpp"

namespace mrf {

inline double smoothnessWeight(const Pixel& p, const Pixel& neighbor, const double& val_p,
                               const double& val_neighbor, const Parameters& params) {
//    return exp(-abs(val_p - val_neighbor));

    return val_p < params.discontinuity_threshold;
}
}
