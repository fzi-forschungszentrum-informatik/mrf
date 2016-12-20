#pragma once
#include <memory>
#include <ostream>
#include <ceres/problem.h>
#include <ceres/solver.h>

namespace mrf {

struct Parameters {

    enum class Neighborhood { four = 4, eight = 8 };
    enum class Initialization { nearest_neighbor, triangles };

    using Ptr = std::shared_ptr<Parameters>;

    inline Parameters(){
    	problem.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    	problem.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    };

    inline static Ptr create() {
        return std::make_shared<Parameters>();
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
        os << "discontinuity threshold: " << p.discontinuity_threshold << std::endl
           << "ks: " << p.ks << std::endl
           << "kd: " << p.kd << std::endl
           << "max_iterations: " << p.max_iterations << std::endl
           << "set_depth_limits: " << p.use_custom_depth_limits << std::endl;
    }

    double ks{1};
    double kd{1};
    double discontinuity_threshold{20};
    Neighborhood neighborhood{Neighborhood::four};
    int max_iterations{20};
    Initialization initialization{Initialization::triangles};
    double radius_normal_estimation{0.5};
    bool use_custom_depth_limits{true};
    double custom_depth_limit_min{0};
    double custom_depth_limit_max{100};

    ceres::Solver::Options solver;
    ceres::Problem::Options problem;
    std::shared_ptr<ceres::LossFunction> loss_function{nullptr};
};
}
