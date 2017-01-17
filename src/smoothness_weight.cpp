#include "smoothness_weight.hpp"

namespace mrf {

double sigmoid(const double& x, const double& alpha, const double& beta) {
    return 0;
}

double smoothnessWeight(const Pixel& p, const Pixel& neighbor, const double& threshold,
                        const double& weight_min,
                        const Parameters::SmoothnessWeighting& smoothness_weighting,
                        const double& alpha, const double& beta) {

    const double diff_abs = (p.val - neighbor.val).norm();
    if (diff_abs < threshold) {
        return 1;
    }

    switch (smoothness_weighting) {
    case Parameters::SmoothnessWeighting::none:
        return 1;
    case Parameters::SmoothnessWeighting::step:
        return weight_min;
    case Parameters::SmoothnessWeighting::linear:
        return std::max(weight_min, 1 - alpha * (diff_abs - threshold));
    case Parameters::SmoothnessWeighting::exponential:
        return std::max(weight_min, std::exp(-alpha * (diff_abs - threshold)));
    case Parameters::SmoothnessWeighting::sigmoid:
        return std::max(weight_min, sigmoid(diff_abs - threshold, alpha, beta));
    }
    return 0;
}
}
