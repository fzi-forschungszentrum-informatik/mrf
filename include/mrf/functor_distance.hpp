#pragma once

#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorDistance {

    static constexpr size_t DimDistance = 1;
    static constexpr size_t DimResidual = 3;
    static constexpr size_t DimRotation = 4;
    static constexpr size_t DimTranslation = 3;

    inline FunctorDistance(const Eigen::Vector3d& p, const Eigen::ParametrizedLine<double, 3>& ray)
            : p_{p}, ray_{ray} {};

    template <typename T>
    inline bool operator()(const T* const depth, const T* const rotation,
                           const T* const translation, T* res) const {
        using namespace Eigen;
        const Affine3<T> tf{util_ceres::fromQuaternionTranslation(rotation, translation)};
        Map<Vector3<T>>(res, DimResidual) = ray_.cast<T>().pointAt(depth[0]) - tf * p_.cast<T>();
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::Vector3d& p,
                                              const Eigen::ParametrizedLine<double, 3>& ray) {
        return new ceres::AutoDiffCostFunction<FunctorDistance, DimResidual, DimDistance, DimRotation,
                                               DimTranslation>(new FunctorDistance(p, ray));
    }

private:
    Eigen::Vector3d p_; ///< 3D world point associated to this pixel
    Eigen::ParametrizedLine<double, 3> ray_;
};
}
