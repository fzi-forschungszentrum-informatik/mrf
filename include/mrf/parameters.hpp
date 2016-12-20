#pragma once
#include <memory>
#include <ostream>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <glog/logging.h>
#include <yaml-cpp/node/node.h>

namespace mrf {

struct Parameters {

    enum class Neighborhood { four = 4, eight = 8 };
    enum class Initialization { none, nearest_neighbor, triangles };
    enum class Limits { none, custom, adaptive };

    using Ptr = std::shared_ptr<Parameters>;

    inline Parameters(const std::string& file_name = std::string()) {
        problem.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        if (file_name.size()) {
            fromConfig(file_name);
        }
    }

    inline static Ptr create(const std::string& file_name = std::string()) {
        return std::make_shared<Parameters>(file_name);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
        os << "discontinuity threshold: " << p.discontinuity_threshold << std::endl
           << "ks: " << p.ks << std::endl
           << "kd: " << p.kd << std::endl
           << "max_iterations: " << p.max_iterations << std::endl
           << "limits: " << static_cast<int>(p.limits) << std::endl;
    }

    double ks{1};
    double kd{1};
    double discontinuity_threshold{0.2};
    int max_iterations{20};
    double radius_normal_estimation{0.5};
    Limits limits{Limits::none};
    double custom_depth_limit_min{0};
    double custom_depth_limit_max{100};

    Initialization initialization{Initialization::none};
    Neighborhood neighborhood{Neighborhood::four};

    ceres::Solver::Options solver;
    ceres::Problem::Options problem;
    std::shared_ptr<ceres::LossFunction> loss_function{nullptr};

private:
    void fromConfig(const std::string& file_name);

    template <typename T>
    inline bool getParam(const YAML::Node& cfg, const std::string& name, T& val) {
        if (cfg[name]) {
            val = cfg[name].as<T>();
            return true;
        }
        return false;
    }
};
}
