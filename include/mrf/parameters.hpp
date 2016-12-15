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

    inline Parameters(){};

    inline static Ptr create() {
        return std::make_shared<Parameters>();
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
        os << "discontinuity threshold: " << p.discontinuity_threshold << std::endl;
        os << "ks: " << p.ks << std::endl;
        os << "kd: " << p.kd << std::endl;
        os << "max_iterations: " << p.max_iterations << std::endl;
        os << "set_depth_limits: " << p.set_depth_limits << std::endl;
    }

    double ks{1};
    double kd{1};
    double discontinuity_threshold{20};
    Neighborhood neighborhood{Neighborhood::four};
    int max_iterations{20};
    Initialization initialization{Initialization::triangles};
    double radius_normal_estimation{0.5};
    bool set_depth_limits{true};

    ceres::Solver::Options solver;
    ceres::Problem::Options problem;
};
}
