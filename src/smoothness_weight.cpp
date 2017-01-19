#include "smoothness_weight.hpp"

#include <cmath>

namespace mrf {

double sigmoid(const double& x, const double& alpha, const double& beta) {
    return 0;
}

double smoothnessWeight(const Pixel& p,
                        const Pixel& neighbor,
                        const double& threshold,
                        const double& weight_min,
                        const Parameters::SmoothnessWeighting& smoothness_weighting,
                        const double& alpha,
                        const double& beta) {
    int w_instance, w_color{1};
    double diff_color, diff_instance{0};
    switch (p.val.rows()) {
    case 1: //> gray image, no instance
        diff_color = std::abs(p.val[0] - neighbor.val[0]);
        w_instance = 0;
        w_color = 1;
        break;
    case 2: //> gray image, plus instance
        diff_color = std::abs(p.val[0] - neighbor.val[0]);
        diff_instance = p.val(1) - neighbor.val(1);
        break;
    case 3: //> color image, no instance
        diff_color = (p.val - neighbor.val).norm();
        w_instance = 0;
        w_color = 1;
        break;
    case 4: //> color image, plus instance
        diff_color = (p.val - neighbor.val).norm();
        diff_instance = p.val(3) - neighbor.val(3);
        break;
    default:
        diff_color = (p.val - neighbor.val).norm();
        w_instance = 0;
        w_color = 1;
        break;
    }
    double color_term, instance_term{0};
    if (std::abs(diff_instance) < 1) {
        instance_term = 1;
    }

    if (diff_color < threshold) {
        color_term = 1;
    } else {
        switch (smoothness_weighting) {
        case Parameters::SmoothnessWeighting::none:
            color_term = 1;
            break;
        case Parameters::SmoothnessWeighting::step:
            color_term = weight_min;
            break;
        case Parameters::SmoothnessWeighting::linear:
            color_term = std::max(weight_min, 1 - alpha * (diff_color - threshold));
            break;
        case Parameters::SmoothnessWeighting::exponential:
            color_term = std::max(weight_min, std::exp(-alpha * (diff_color - threshold)));
            break;
        case Parameters::SmoothnessWeighting::sigmoid:
            color_term = std::max(weight_min, sigmoid(diff_color - threshold, alpha, beta));
            break;
        }
    }


    return (color_term * w_color + instance_term * w_instance) / (w_instance + w_color);
}
}
