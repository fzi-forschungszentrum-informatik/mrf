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
    enum class Initialization {
        none = 0,
        mean_depth = 1,
        nearest_neighbor = 2,
        triangles = 3,
        weighted_neighbor = 4
    };
    enum class Limits { none, custom, adaptive };
    enum class SmoothnessWeighting { none = 0, step = 1, linear = 2, exponential = 3, sigmoid = 4 };
    enum class CropMode { none = 0, min_max, box };

    static constexpr char del = ','; ///< Delimiter

    Parameters(const std::string& = std::string());
    static std::string header();
    std::string toString() const;
    inline friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
        return os << p.toString();
    }

    ceres::LossFunction* createLossFunction() const;

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
    bool use_functor_normal{false};
    bool use_functor_normal_distance{true};
    bool use_functor_smoothness_normal{false};
    bool use_functor_smoothness_distance{false};
    bool pin_normals{false};
    bool pin_distances{false};
    double loss_function_scale{1};
    Initialization initialization{Initialization::mean_depth};
    Neighborhood neighborhood{Neighborhood::eight};
    bool estimate_covariances{false};
    ceres::Problem::Options problem;
    std::string loss_function{"trivial"};
    CropMode crop_mode{CropMode::none};
    bool use_covariance_filter{false};
    double covariance_filter_treshold{0.1};
    double sigmoid_scale{0.5};

    int box_cropping_row_min{0};
    int box_cropping_row_max{5000};
    int box_cropping_col_min{0};
    int box_cropping_col_max{5000};

private:
    void fromConfig(const std::string&);

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
