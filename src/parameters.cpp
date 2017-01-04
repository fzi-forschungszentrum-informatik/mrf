#include "parameters.hpp"

#include <yaml-cpp/yaml.h>

namespace mrf {

void Parameters::fromConfig(const std::string& file_name) {
    const YAML::Node cfg{YAML::LoadFile(file_name)};

    std::string tmp;

    getParam(cfg, "ks", ks);
    getParam(cfg, "kd", kd);
    getParam(cfg, "kn", kn);
    getParam(cfg, "discontinuity_threshold", discontinuity_threshold);
    getParam(cfg, "smoothness_rate", smoothness_rate);
    getParam(cfg, "max_num_iterations", solver.max_num_iterations);
    getParam(cfg, "radius_normal_estimation", radius_normal_estimation);
    getParam(cfg, "neighbor_search", neighbor_search);

    getParam(cfg, "estimate_normals", estimate_normals);
    getParam(cfg, "use_functor_normal_distance", use_functor_normal_distance);
    getParam(cfg, "use_functor_smoothness_normal", use_functor_smoothness_normal);
    getParam(cfg, "use_functor_normal", use_functor_normal);
    getParam(cfg, "use_functor_distance", use_functor_distance);
    getParam(cfg, "use_functor_smoothness_distance", use_functor_smoothness_distance);

    getParam(cfg, "pin_normals", pin_normals);
    getParam(cfg, "pin_distances", pin_distances);

    if (getParam(cfg, "limits", tmp)) {
        if (tmp == "none") {
            limits = Limits::none;
        } else if (tmp == "custom") {
            limits = Limits::custom;
        } else if (tmp == "adaptive") {
            limits = Limits::adaptive;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }
    getParam(cfg, "custom_depth_limit_min", custom_depth_limit_min);
    getParam(cfg, "custom_depth_limit_max", custom_depth_limit_max);

    if (getParam(cfg, "initialization", tmp)) {
        if (tmp == "none") {
            initialization = Initialization::none;
        } else if (tmp == "nearest_neighbor") {
            initialization = Initialization::nearest_neighbor;
        } else if (tmp == "mean_depth") {
            initialization = Initialization::mean_depth;
        } else if (tmp == "triangles") {
            initialization = Initialization::triangles;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }

    if (getParam(cfg, "neighborhood", tmp)) {
        if (tmp == "four") {
            neighborhood = Neighborhood::four;
        } else if (tmp == "eight") {
            neighborhood = Neighborhood::eight;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }

    getParam(cfg, "loss_function_scale", loss_function_scale);
    if (getParam(cfg, "loss_function", tmp)) {
        if (tmp == "trivial") {
            loss_function = std::make_shared<ceres::TrivialLoss>();
        } else if (tmp == "huber") {
            loss_function = std::make_shared<ceres::HuberLoss>(loss_function_scale);
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }
}
}
