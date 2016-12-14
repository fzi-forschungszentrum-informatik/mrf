#pragma once

#include <memory>
#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

class FunctorDistance {

    using Ptr = std::shared_ptr<FunctorDistance>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimRotation = 4;
    static constexpr size_t DimTranslation = 3;

public:
    inline FunctorDistance(const Eigen::Vector3d& p, const double& w) : p_{p}, w_{w} {};

    template <typename T>
    inline bool operator()(const T* const depth, const T* const rotation,
                           const T* const translation, T* res) const {
        const Eigen::Affine3<T> tf{util_ceres::fromQuaternionTranslation(rotation, translation)};
        res[0] = T(w_) * (depth[0] - (tf * p_.cast<T>()).norm());
        return true;
    }

    inline static Ptr create(const Eigen::Vector3d& p, const double& w) {
        return std::make_shared<FunctorDistance>(p, w);
    }

    inline ceres::CostFunction* toCeres() {
        return new ceres::AutoDiffCostFunction<FunctorDistance, DimResidual, DimDepth, DimRotation,
                                               DimTranslation>(this);
    }

    inline void setPoint(const Eigen::Vector3d& p) {
        p_ = p;
    }
    inline void setWeight(const double w) {
        w_ = w;
    }

private:
    Eigen::Vector3d p_; ///< 3D world point associated to this pixel
    const double w_;    ///< Weight
};
