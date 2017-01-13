#pragma once

#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorNormal {

    static constexpr size_t DimNormal = 3;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormal(const Eigen::Vector3d& n) : n_{n} {};

    template <typename T>
    inline bool operator()(const T* const n_ceres, const T* const rot_ceres, T* res_ceres) const {
        using namespace Eigen;
        const Map<const Vector3<T>> n(n_ceres);
        res_ceres[0] = T(1) - n.dot(util_ceres::fromQuaternion(rot_ceres) * n_.cast<T>());
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::Vector3d& n) {
        return new ceres::AutoDiffCostFunction<FunctorNormal, DimResidual, DimNormal, DimRotation>(
            new FunctorNormal(n));
    }

private:
    Eigen::Vector3d n_; ///< 3D world point normal associated to this pixel
};
}
