#pragma once

#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorSmoothnessNormal {

    static constexpr size_t DimNormal = 3;
    static constexpr size_t DimResidual = 1;

    template <typename T>
    inline bool operator()(const T* const n_this_ceres,
                           const T* const n_nn_ceres,
                           T* res_ceres) const {
        using namespace Eigen;
        res_ceres[0] = Map<const Vector3<T>>{n_this_ceres}.dot(Map<const Vector3<T>>{n_nn_ceres}) -
                       static_cast<T>(1);
        return true;
    }

    inline static ceres::CostFunction* create() {
        return new ceres::
            AutoDiffCostFunction<FunctorSmoothnessNormal, DimResidual, DimNormal, DimNormal>(
                new FunctorSmoothnessNormal());
    }
};
}
