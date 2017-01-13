#pragma once

#include <ceres/autodiff_cost_function.h>
#include <Eigen/src/Geometry/Hyperplane.h>
#include <Eigen/src/Geometry/ParametrizedLine.h>

namespace mrf {

struct FunctorNormalDistance {

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimNormal = 3;

    inline FunctorNormalDistance(const Eigen::ParametrizedLine<double, 3>& ray_this,
                                 const Eigen::ParametrizedLine<double, 3>& ray_nn)
            : ray_this_{ray_this}, ray_nn_{ray_nn} {};

    template <typename T>
    inline bool operator()(const T* const depth_this, const T* const depth_nn,
                           const T* const normal_this, T* res) const {
        using namespace Eigen;
        const Hyperplane<T, 3> plane_this(Map<const Vector3<T>>(normal_this),
                                          ray_this_.cast<T>().pointAt(depth_this[0]));
        const Vector3<T> p_nn{ray_nn_.cast<T>().pointAt(depth_nn[0])};
        res[0] = plane_this.signedDistance(p_nn);
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::ParametrizedLine<double, 3>& ray_this,
                                              const Eigen::ParametrizedLine<double, 3>& ray_nn) {
        return new ceres::AutoDiffCostFunction<FunctorNormalDistance, DimResidual, DimDepth,
                                               DimDepth, DimNormal>(
            new FunctorNormalDistance(ray_this, ray_nn));
    }

private:
    const Eigen::ParametrizedLine<double, 3> ray_this_, ray_nn_;
};
}
