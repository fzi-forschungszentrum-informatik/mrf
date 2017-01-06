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
    enum class SmoothnessWeighting {none = 0, step, exponential, sigmoid};

    inline Parameters(const std::string& file_name = std::string()) {
        problem.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        solver.max_num_iterations = 25;
        solver.minimizer_progress_to_stdout = true;
        solver.num_threads = 8;
        solver.num_linear_solver_threads = 8;
        solver.max_solver_time_in_seconds = 600;
        solver.use_inner_iterations = true;
        solver.use_nonmonotonic_steps = true;
        solver.function_tolerance = 1e-5;

        if (file_name.size()) {
            fromConfig(file_name);
        }
        LOG(INFO) << *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Parameters& p);

    double ks{1};
    double kd{1};
    double kn{1};
    double discontinuity_threshold{0.1};
    double smoothness_rate{50};
    double smoothness_weight_min{0.001};
    double radius_normal_estimation{0.5};
    Limits limits{Limits::none};
    double custom_depth_limit_min{0};
    double custom_depth_limit_max{100};
    SmoothnessWeighting smoothness_weighting{SmoothnessWeighting::step};
    int neighbor_search{20};

    bool estimate_normals{true};
    bool use_functor_normal_distance{true};
    bool use_functor_normal{true};
    bool use_functor_distance{true};
    bool use_functor_smoothness_normal{true};
    bool use_functor_smoothness_distance{false};

    bool pin_normals{false};
    bool pin_distances{false};

    double loss_function_scale{1};

    bool estimate_covariances{false};

    Initialization initialization{Initialization::mean_depth};
    Neighborhood neighborhood{Neighborhood::eight};

    ceres::Solver::Options solver;
    ceres::Problem::Options problem;
    std::shared_ptr<ceres::LossFunction> loss_function{new ceres::TrivialLoss};

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
