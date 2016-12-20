#pragma once

#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

inline double smoothnessWeight(const Pixel& p, const Pixel& neighbor, const Parameters& params) {
    //    return exp(-abs(val_p - val_neighbor));

    return std::abs(p.val - neighbor.val) < params.discontinuity_threshold;
}
}
