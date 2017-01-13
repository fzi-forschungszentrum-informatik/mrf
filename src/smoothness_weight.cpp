#include "smoothness_weight.hpp"

#include "math.hpp"

namespace mrf {

double smoothnessWeight(const Pixel& p,
                        const double& threshold,
                        const double& weight_min,
                        const Parameters::SmoothnessWeighting& smoothness_weighting,
                        const double& alpha,
                        const double& beta) {
    if (p.val < threshold) {
        return 1;
    }

    switch (smoothness_weighting) {
    case Parameters::SmoothnessWeighting::none:
        return 1;
    case Parameters::SmoothnessWeighting::step:
        return weight_min;
    case Parameters::SmoothnessWeighting::linear:
        return std::max(weight_min, 1 - alpha * (p.val - threshold));
    case Parameters::SmoothnessWeighting::exponential:
        return std::max(weight_min, std::exp(-alpha * (p.val - threshold)));
    case Parameters::SmoothnessWeighting::sigmoid:
        return std::max(weight_min, sigmoid(p.val - threshold, alpha, beta));
    }
    return 0;
}
}
