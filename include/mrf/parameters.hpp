#pragma once

#include <memory>
#include <ostream>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace mrf {

struct Parameters {

    enum class Neighborhood { two = 2, four = 4, eight = 8 };
    enum class Initialization { none, nearest_neighbor, triangles, mean_depth };
    enum class Limits { none, custom, adaptive };
    enum class SmoothnessWeighting { none = 0, step, linear, exponential, sigmoid };
    enum class CropMode { none = 0, min_max };

    inline Parameters(const std::string& file_name = std::string()) {
        using namespace ceres;
        problem.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
        problem.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
        solver.max_num_iterations = 50;
        solver.minimizer_progress_to_stdout = true;
        solver.num_threads = 8;
        solver.num_linear_solver_threads = 8;
        solver.max_solver_time_in_seconds = 600;
        solver.use_inner_iterations = true;
        solver.use_nonmonotonic_steps = true;
        solver.function_tolerance = 1e-5;
        solver.minimizer_type = MinimizerType::TRUST_REGION;
        solver.linear_solver_type = LinearSolverType::CGNR;
        solver.sparse_linear_algebra_library_type = SparseLinearAlgebraLibraryType::SUITE_SPARSE;

        if (file_name.size())
            fromConfig(file_name);

        LOG(INFO) << header();
        LOG(INFO) << *this;
    }

    static constexpr char del = ','; ///< Delimiter
    inline static std::string header() {
        std::ostringstream oss;
        // clang-format off
        oss << "ks" << del
        	<< "kd" << del
			<< "kn" << del
			<< "discontinuity_threshold" << del
            << "limits" << del
			<< "discontinuity_threshold"<< del
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
			<< "covariance_filter_treshold";
        // clang-format on
        return oss.str();
    }
    inline friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
        // clang-format off
        return os << p.ks << del
        		<< p.kd << del
				<< p.kn << del
				<< p.discontinuity_threshold << del
				<< static_cast<int>(p.limits) << del
				<< p.custom_depth_limit_min << del
                << p.custom_depth_limit_max << del
				<< static_cast<int>(p.smoothness_weighting) << del
				<< p.smoothness_rate << del
				<< p.smoothness_weight_min << del
                << p.radius_normal_estimation << del
				<< p.neighbor_search << del
                << p.solver.max_num_iterations << del
				<< p.solver.minimizer_progress_to_stdout << del
				<< p.solver.num_threads << del
				<< p.solver.num_linear_solver_threads << del
                << p.solver.max_solver_time_in_seconds << del
				<< p.solver.use_inner_iterations << del
				<< p.solver.use_nonmonotonic_steps << del
				<< p.estimate_normals << del
                << p.use_functor_distance << del
				<< p.use_functor_normal << del
                << p.use_functor_normal_distance << del
				<< p.use_functor_smoothness_normal << del
                << p.use_functor_smoothness_distance << del
				<< p.pin_normals << del
                << p.pin_distances << del
				<< p.loss_function_scale << del
                << static_cast<int>(p.initialization) << del
				<< static_cast<int>(p.neighborhood) << del
				<< p.estimate_covariances << del
				<< static_cast<int>(p.crop_mode) << del
				<< p.use_covariance_filter << del
				<< p.covariance_filter_treshold;
        // clang-format on
    }

    double ks{1};
    double kd{1};
    double kn{1};
    double discontinuity_threshold{0.1};
    Limits limits{Limits::none};
    double custom_depth_limit_min{0};
    double custom_depth_limit_max{100};
    SmoothnessWeighting smoothness_weighting{SmoothnessWeighting::step};
    double smoothness_rate{50};
    double smoothness_weight_min{0.001};
    double radius_normal_estimation{0.5};
    int neighbor_search{20};
    ceres::Solver::Options solver;
    bool estimate_normals{true};
    bool use_functor_distance{true};
    bool use_functor_normal{true};
    bool use_functor_normal_distance{true};
    bool use_functor_smoothness_normal{true};
    bool use_functor_smoothness_distance{false};
    bool pin_normals{false};
    bool pin_distances{false};
    double loss_function_scale{1};
    Initialization initialization{Initialization::mean_depth};
    Neighborhood neighborhood{Neighborhood::eight};
    bool estimate_covariances{false};
    ceres::Problem::Options problem;
    std::shared_ptr<ceres::LossFunction> loss_function{new ceres::TrivialLoss};
    CropMode crop_mode{CropMode::none};
    bool use_covariance_filter{false};
    double covariance_filter_treshold{0.1};

private:
    void fromConfig(const std::string& file_name);

    template <typename T>
    inline bool getParam(const YAML::Node& cfg, const std::string& name, T& val) {
        if (cfg[name]) {
            val = cfg[name].as<T>();
            return true;
        }
        LOG(WARNING) << "Parameter " << name << " not available.";
        return false;
    }
};
}
