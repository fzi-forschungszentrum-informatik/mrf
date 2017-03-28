#pragma once

#include <memory>
#include <ostream>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace mrf {

/** @brief Parameters for an optimization. */
struct Parameters {
    /** @brief Method to get initial values for all pixels. */
    enum class Initialization {
        none = 0,             ///< No initial value
        mean_depth = 1,       ///< Interpolate between neighbors
        nearest_neighbor = 2, ///< Set depth to nearest neighbor
        triangles = 3,        ///< Calculate from triangle between neighbors
        weighted_neighbor = 4 ///< Weight influence of neighbors by distance and smoothness
    };
    /** @brief Type of limits for minimum and maximum depth */
    enum class Limits {
        none,    ///< No constraints
        custom,  ///< Custom defined limits
        adaptive ///< Derived from dataset
    };
    /** @brief Method for the smoothness weighting between pixels. */
    enum class SmoothnessWeighting {
        none = 0,        ///< Always 1
        step = 1,        ///< Parametrized step
        linear = 2,      ///< Parametrized linear
        exponential = 3, ///< Parametrized exponential
        sigmoid = 4      ///< Not implemented
    };
    /** @brief  */
    enum class CropMode {
        none = 0, ///< Use all pixels
        min_max,  ///< Derive min, max values from image
        box       ///< Set a custom box
    };

    static constexpr char del = ','; ///< Delimiter

    Parameters(const std::string& = std::string());
    static std::string header();
    std::string toString() const;
    inline friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
        return os << p.toString();
    }

    ceres::Solver::Options solver;
    ceres::Problem::Options problem;
    ceres::LossFunction* createLossFunction() const;

    double ks{1};                       ///< Smoothness weight
    double kd{1};                       ///< Distance weight
    double kn{1};                       ///< TODO wtf?
    Limits limits{Limits::none};        ///< Min, max depth limit method
    double custom_depth_limit_min{0};   ///< Minimum depth if Limits::custom
    double custom_depth_limit_max{100}; ///< Maximum depth if Limits::custom
    SmoothnessWeighting smoothness_weighting{
        SmoothnessWeighting::step};      ///< Smoothness weighting scaling function type
    double discontinuity_threshold{0.1}; ///< Smoothness delta before applying scaling functions
    double smoothness_rate{50};          ///< Decrease rate
    double smoothness_weight_min{
        0.001}; ///< Lower threshold for the smoothness weight to prevent isolated areas
    double sigmoid_scale{0.5};   ///< Sigmoid scale
    bool estimate_normals{true}; ///< Perform normal estimation. Choose no if normals are provided.
    double radius_normal_estimation{0.5}; ///< Radius for normal estimation
    int neighbor_search{20}; ///< Number of neighbors to search for during initialization
    Initialization initialization{Initialization::mean_depth}; ///< Initialization method
    CropMode crop_mode{CropMode::none};   ///< Crop mode to reduce the amount of data
    int box_cropping_row_min{0};          ///< Minimum row value for box cropping
    int box_cropping_row_max{5000};       ///< Maximum row value for box cropping
    int box_cropping_col_min{0};          ///< Minimum col value for box cropping
    int box_cropping_col_max{5000};       ///< Maximum col value for box cropping
    std::string loss_function{"trivial"}; ///< Type of loss function
    double loss_function_scale{1};        ///< Scale of loss function

    bool use_functor_distance{true};
    bool use_functor_normal{false};
    bool use_functor_normal_distance{true};
    bool use_functor_smoothness_normal{false};
    bool use_functor_smoothness_distance{false};

    bool pin_normals{false}; ///< TODO wtf?
    bool pin_distances{
        false}; ///< Pin depth estimation to actual value for pixel with range measurments
    bool pin_transform{true};
    bool estimate_covariances{false};        ///< Calculate covariances in postprocessing
    bool use_covariance_filter{false};       ///< Perform covariance based filter in postprocessing
    double covariance_filter_threshold{0.1}; ///< Threshold for covariance filter

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
