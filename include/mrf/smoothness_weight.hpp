#pragma once

#include "pixel.hpp"

namespace mrf {

inline double smoothnessWeight(const Pixel& p, const Pixel& neighbor, const double& val_p,
                               const double& val_neighbor) {
//    return exp(-abs(val_p - val_neighbor));

    return val_p < 0.7;
}
}
