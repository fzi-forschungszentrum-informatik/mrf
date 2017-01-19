#pragma once

#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

double smoothnessWeight(const Pixel& p,
                        const Pixel& neighbor,
                        const double& threshold,
                        const double& weight_min,
                        const Parameters::SmoothnessWeighting&,
                        const double& alpha = -1,
                        const double& beta = -1);
}
