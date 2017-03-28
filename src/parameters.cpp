#include <yaml-cpp/yaml.h>

#include "parameters.hpp"

namespace mrf {

Parameters::Parameters(const std::string& file_name) {
    using namespace ceres;
    solver.max_num_iterations = 50;
    solver.minimizer_progress_to_stdout = true;
    solver.num_threads = 8;
    solver.num_linear_solver_threads = 8;
    solver.max_solver_time_in_seconds = 600;

    if (file_name.size())
        fromConfig(file_name);

    LOG(INFO) << header() << std::endl << toString();
}

std::string Parameters::header() {
    std::ostringstream oss;
    // clang-format off
	oss << "ks" << del
		<< "kd" << del
		<< "kn" << del
		<< "discontinuity_threshold" << del
		<< "limits" << del
		<< "custom_depth_limit_min" << del
		<< "custom_depth_limit_max" << del
		<< "smoothness_weighting" << del
		<< "smoothness_rate" << del
		<< "smoothness_weight_min" << del
		<< "radius_normal_estimation" << del
		<< "neighbor_search" << del
		<< "max_iterations" << del
		<< "minimizer_progress_to_stdout" << del
		<< "num_threads" << del
		<< "num_linear_solver_threads" << del
		<< "max_solver_time_in_seconds" << del
		<< "use_inner_iterations" << del
		<< "use_nonmonotonic_steps" << del
		<< "estimate_normals" << del
		<< "use_functor_distance" << del
		<< "use_functor_normal" << del
		<< "use_functor_normal_distance" << del
		<< "use_functor_smoothness_normal" << del
		<< "use_functor_smoothness_distance" << del
		<< "pin_normals" << del
		<< "pin_distances" << del
		<< "loss_function_scale" << del
		<< "initialization" << del
		<< "neighborhood" << del
		<< "estimate_covariances" << del
		<< "crop_mode" << del
		<< "use_covariance_filter" << del
        << "covariance_filter_threshold" << del
		<< "sigmoid_scale" << del
		<< "box_cropping_row_min" << del
		<< "box_cropping_row_max" << del
		<< "box_cropping_col_min" << del
		<< "box_cropping_col_max";
    // clang-format on
    return oss.str();
}

std::string Parameters::toString() const {
    std::ostringstream oss;
    // clang-format off
	oss << ks << del
		<< kd << del
		<< kn << del
		<< discontinuity_threshold << del
		<< static_cast<int>(limits) << del
		<< custom_depth_limit_min << del
		<< custom_depth_limit_max << del
		<< static_cast<int>(smoothness_weighting) << del
		<< smoothness_rate << del
		<< smoothness_weight_min << del
		<< radius_normal_estimation << del
		<< neighbor_search << del
		<< solver.max_num_iterations << del
		<< solver.minimizer_progress_to_stdout << del
		<< solver.num_threads << del
		<< solver.num_linear_solver_threads << del
		<< solver.max_solver_time_in_seconds << del
		<< solver.use_inner_iterations << del
		<< solver.use_nonmonotonic_steps << del
		<< estimate_normals << del
		<< use_functor_distance << del
		<< use_functor_normal << del
		<< use_functor_normal_distance << del
		<< use_functor_smoothness_normal << del
		<< use_functor_smoothness_distance << del
		<< pin_normals << del
		<< pin_distances << del
		<< loss_function_scale << del
        << static_cast<int>(initialization) << del
		<< estimate_covariances << del
		<< static_cast<int>(crop_mode) << del
		<< use_covariance_filter << del
        << covariance_filter_threshold << del
		<< sigmoid_scale << del
		<< box_cropping_row_min << del
		<< box_cropping_row_max << del
		<< box_cropping_col_min << del
		<< box_cropping_col_max;
    // clang-format on
    return oss.str();
}

void Parameters::fromConfig(const std::string& file_name) {
    const YAML::Node cfg{YAML::LoadFile(file_name)};

    std::string tmp;

    getParam(cfg, "ks", ks);
    getParam(cfg, "kd", kd);
    getParam(cfg, "kn", kn);

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
            LOG(INFO) << "Smoothness weighting: none";
        } else if (tmp == "step") {
            smoothness_weighting = SmoothnessWeighting::step;
            LOG(INFO) << "Smoothness weighting: step";
        } else if (tmp == "linear") {
            smoothness_weighting = SmoothnessWeighting::linear;
            LOG(INFO) << "Smoothness weighting: linear";
        } else if (tmp == "exponential") {
            smoothness_weighting = SmoothnessWeighting::exponential;
            LOG(INFO) << "Smoothness weighting: exponential";
        } else if (tmp == "sigmoid") {
            smoothness_weighting = SmoothnessWeighting::sigmoid;
            LOG(INFO) << "Smoothness weighting: sigmoid";
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }
    getParam(cfg, "discontinuity_threshold", discontinuity_threshold);
    getParam(cfg, "smoothness_rate", smoothness_rate);
    getParam(cfg, "smoothness_weight_min", smoothness_weight_min);
    getParam(cfg, "sigmoid_scale", sigmoid_scale);
    getParam(cfg, "estimate_normals", estimate_normals);
    getParam(cfg, "radius_normal_estimation", radius_normal_estimation);
    getParam(cfg, "neighbor_search", neighbor_search);

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

    if (getParam(cfg, "crop_mode", tmp)) {
        if (tmp == "none") {
            crop_mode = CropMode::none;
        } else if (tmp == "min_max") {
            crop_mode = CropMode::min_max;
        } else if (tmp == "box") {
            crop_mode = CropMode::box;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }
    getParam(cfg, "box_cropping_row_min", box_cropping_row_min);
    getParam(cfg, "box_cropping_row_max", box_cropping_row_max);
    getParam(cfg, "box_cropping_col_min", box_cropping_col_min);
    getParam(cfg, "box_cropping_col_max", box_cropping_col_max);

    getParam(cfg, "loss_function", loss_function);
    getParam(cfg, "loss_function_scale", loss_function_scale);

    getParam(cfg, "max_num_iterations", solver.max_num_iterations);
    getParam(cfg, "minimizer_progress_to_stdout", solver.minimizer_progress_to_stdout);
    getParam(cfg, "num_threads", solver.num_threads);
    getParam(cfg, "num_linear_solver_threads", solver.num_linear_solver_threads);
    getParam(cfg, "max_solver_time_in_seconds", solver.max_solver_time_in_seconds);
    getParam(cfg, "use_inner_iterations", solver.use_inner_iterations);
    getParam(cfg, "use_nonmonotonic_steps", solver.use_nonmonotonic_steps);
    getParam(cfg, "function_tolerance", solver.function_tolerance);

    getParam(cfg, "use_functor_normal_distance", use_functor_normal_distance);
    getParam(cfg, "use_functor_smoothness_normal", use_functor_smoothness_normal);
    getParam(cfg, "use_functor_normal", use_functor_normal);
    getParam(cfg, "use_functor_distance", use_functor_distance);
    getParam(cfg, "use_functor_smoothness_distance", use_functor_smoothness_distance);

    getParam(cfg, "pin_normals", pin_normals);
    getParam(cfg, "pin_distances", pin_distances);
    getParam(cfg, "pin_transform", pin_transform);

    getParam(cfg, "estimate_covariances", estimate_covariances);
    getParam(cfg, "use_covariance_filter", use_covariance_filter);
    getParam(cfg, "covariance_filter_threshold", covariance_filter_threshold);
}

ceres::LossFunction* Parameters::createLossFunction() const {
    if (loss_function == "huber") {
        return new ceres::HuberLoss(loss_function_scale);
    } else if (loss_function == "cauchy") {
        return new ceres::CauchyLoss(loss_function_scale);
    }
    return new ceres::TrivialLoss;
}
}
