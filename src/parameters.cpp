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
    getParam(cfg, "smoothness_weight_min", smoothness_weight_min);
    getParam(cfg, "radius_normal_estimation", radius_normal_estimation);
    getParam(cfg, "neighbor_search", neighbor_search);

    getParam(cfg, "max_num_iterations", solver.max_num_iterations);
    getParam(cfg, "minimizer_progress_to_stdout", solver.minimizer_progress_to_stdout);
    getParam(cfg, "num_threads", solver.num_threads);
    getParam(cfg, "num_linear_solver_threads", solver.num_linear_solver_threads);
    getParam(cfg, "max_solver_time_in_seconds", solver.max_solver_time_in_seconds);
    getParam(cfg, "use_inner_iterations", solver.use_inner_iterations);
    getParam(cfg, "use_nonmonotonic_steps", solver.use_nonmonotonic_steps);
    getParam(cfg, "function_tolerance", solver.function_tolerance);

    getParam(cfg, "estimate_normals", estimate_normals);
    getParam(cfg, "use_functor_normal_distance", use_functor_normal_distance);
    getParam(cfg, "use_functor_smoothness_normal", use_functor_smoothness_normal);
    getParam(cfg, "use_functor_normal", use_functor_normal);
    getParam(cfg, "use_functor_distance", use_functor_distance);
    getParam(cfg, "use_functor_smoothness_distance", use_functor_smoothness_distance);

    getParam(cfg, "pin_normals", pin_normals);
    getParam(cfg, "pin_distances", pin_distances);

    getParam(cfg, "estimate_covariances", estimate_covariances);

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

    if (getParam(cfg, "smoothness_weighting", tmp)) {
        if (tmp == "none") {
            smoothness_weighting = SmoothnessWeighting::none;
        } else if (tmp == "step") {
            smoothness_weighting = SmoothnessWeighting::step;
        } else if (tmp == "linear") {
            smoothness_weighting = SmoothnessWeighting::linear;
        } else if (tmp == "exponential") {
            smoothness_weighting = SmoothnessWeighting::exponential;
        } else if (tmp == "sigmoid") {
            smoothness_weighting = SmoothnessWeighting::sigmoid;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }

    if (getParam(cfg, "initialization", tmp)) {
        if (tmp == "none") {
            initialization = Initialization::none;
        } else if (tmp == "nearest_neighbor") {
            initialization = Initialization::nearest_neighbor;
        } else if (tmp == "mean_depth") {
            initialization = Initialization::mean_depth;
        } else if (tmp == "triangles") {
            initialization = Initialization::triangles;
        } else if (tmp == "weighted_neighbor") {
            initialization = Initialization::weighted_neighbor;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }

    if (getParam(cfg, "neighborhood", tmp)) {
        if (tmp == "two") {
            neighborhood = Neighborhood::two;
        } else if (tmp == "four") {
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
        } else if (tmp == "cauchy") {
            loss_function = std::make_shared<ceres::CauchyLoss>(loss_function_scale);
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }

    if (getParam(cfg, "crop_mode", tmp)) {
        if (tmp == "none") {
            crop_mode = CropMode::none;
        } else if (tmp == "min_max") {
            crop_mode = CropMode::min_max;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }
    getParam(cfg, "use_covariance_filter", use_covariance_filter);
    getParam(cfg, "covariance_filter_treshold", covariance_filter_treshold);
}
}
