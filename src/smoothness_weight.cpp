#include "smoothness_weight.hpp"

#include <cmath>

namespace mrf {

double sigmoid(const double& x, const double& alpha, const double& beta) {
    return 0;
}

double smoothnessWeight(const Pixel& p,
                        const Pixel& neighbor,
                        const double& threshold,
                        const double& w_min,
                        const Parameters::SmoothnessWeighting& smoothness_weighting,
                        const double& alpha,
                        const double& beta) {
    double diff_color, diff_instance{0};
    switch (p.val.rows()) {
    case 1: ///> gray image, no instance
        diff_color = std::abs(p.val[0] - neighbor.val[0]);
        break;
    case 2: ///> gray image, plus instance
        diff_color = std::abs(p.val[0] - neighbor.val[0]);
        diff_instance = p.val[1] - neighbor.val[1];
        break;
    case 3: ///> color image, no instance
        diff_color = (p.val - neighbor.val).norm();
        break;
    case 4: ///> color image, plus instance
        diff_color = (p.val.topRows<3>() - neighbor.val.topRows<3>()).norm();
        diff_instance = p.val[3] - neighbor.val[3];
        //        diff_instance = 1;
        break;
    default:
        diff_color = (p.val - neighbor.val).norm();
        break;
    }
    double instance_term{1};
    if (std::abs(diff_instance) > 0) {
        instance_term = w_min;
        //        instance_term = std::max<double>(p.val[3] * neighbor.val[3], w_min);
    }

    double color_term{1};
    if (diff_color > threshold) {
        switch (smoothness_weighting) {
        case Parameters::SmoothnessWeighting::none:
            break;
        case Parameters::SmoothnessWeighting::step:
            color_term = w_min;
            break;
        case Parameters::SmoothnessWeighting::linear:
            color_term = std::max(w_min, 1 - alpha * (diff_color - threshold));
            break;
        case Parameters::SmoothnessWeighting::exponential:
            color_term = std::max(w_min, std::exp(-alpha * (diff_color - threshold)));
            break;
        case Parameters::SmoothnessWeighting::sigmoid:
            color_term = std::max(w_min, sigmoid(diff_color - threshold, alpha, beta));
            break;
        }
    }
    return color_term * instance_term;
}
}
