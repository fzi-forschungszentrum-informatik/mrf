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

    enum class Neighborhood { four = 4, eight = 8 };
    enum class Initialization { none, nearest_neighbor, triangles, mean_depth };
    enum class Limits { none, custom, adaptive };

    inline Parameters(const std::string& file_name = std::string()) {
        problem.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

        solver.max_num_iterations = 20;

        if (file_name.size()) {
            fromConfig(file_name);
        }
        LOG(INFO) << *this;
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
        return os << "discontinuity threshold: " << p.discontinuity_threshold << std::endl
                  << "ks: " << p.ks << std::endl
                  << "kd: " << p.kd << std::endl
                  << "kn: " << p.kn << std::endl
                  << "max_iterations: " << p.solver.max_num_iterations << std::endl
                  << "limits: " << static_cast<int>(p.limits) << std::endl
				  << "smoothness_rate: " << p.smoothness_rate << std::endl
				  << "radius_normal_estimation: " << p.radius_normal_estimation << std::endl
				  << "custom_depth_limit_min: " << p.custom_depth_limit_min << std::endl
				  << "custom_depth_limit_max: " << p.custom_depth_limit_max << std::endl
				  << "neighbor_search: " << p.neighbor_search << std::endl
				  << "estimate_normals: " << p.estimate_normals << std::endl
				  << "use_functor_normal_distance: " << p.use_functor_normal_distance << std::endl
				  << "use_functor_smoothness_normal: " << p.use_functor_smoothness_normal << std::endl
				  << "use_functor_normal: " << p.use_functor_normal << std::endl
				  << "use_functor_distance: " << p.use_functor_distance << std::endl
				  << "use_functor_smoothness_distance: " << p.use_functor_smoothness_distance << std::endl
				  << "pin_normals: " << p.pin_normals << std::endl
				  << "pin_distances: " << p.pin_distances << std::endl
				  << "loss_function_scale: " << p.loss_function_scale << std::endl
                  << "initialization: " << static_cast<int>(p.initialization) << std::endl
				  << "neighborhood: " << static_cast<int>(p.neighborhood) << std::endl;
    }

    double ks{2};
    double kd{1};
    double kn{1};
    double discontinuity_threshold{0.2};
    double smoothness_rate{2};
    double radius_normal_estimation{0.5};
    Limits limits{Limits::none};
    double custom_depth_limit_min{0};
    double custom_depth_limit_max{100};
    int neighbor_search{6};

    bool estimate_normals{true};
    bool use_functor_normal_distance{true};
    bool use_functor_smoothness_normal{true};
    bool use_functor_normal{true};
    bool use_functor_distance{true};
    bool use_functor_smoothness_distance{true};

    bool pin_normals{false};
    bool pin_distances{false};

    double loss_function_scale{1};

    Initialization initialization{Initialization::none};
    Neighborhood neighborhood{Neighborhood::four};

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
