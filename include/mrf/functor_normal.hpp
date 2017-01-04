#pragma once

#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorNormal {

    static constexpr size_t DimNormal = 3;
    static constexpr size_t DimResidual = 3;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormal(const Eigen::Vector3d& n) : n_{n} {};

    template <typename T>
    inline bool operator()(const T* const normal, const T* const rotation, T* res) const {
        using namespace Eigen;
        Map<Vector3<T>>(res, DimResidual) =
            Map<const Vector3<T>>(normal) - util_ceres::fromQuaternion(rotation) * n_.cast<T>();
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
