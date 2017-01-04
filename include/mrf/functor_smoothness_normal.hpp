#pragma once

#include <ceres/autodiff_cost_function.h>

namespace mrf {

struct FunctorSmoothnessNormal {

    static constexpr size_t DimNormal = 3;
    static constexpr size_t DimResidual = 3;

    template <typename T>
    inline bool operator()(const T* const n_this, const T* const n_nn, T* res) const {
        res[0] = n_this[0] - n_nn[0];
        res[1] = n_this[1] - n_nn[1];
        res[2] = n_this[2] - n_nn[2];
        return true;
    }

    inline static ceres::CostFunction* create() {
        return new ceres::AutoDiffCostFunction<FunctorSmoothnessNormal, DimResidual, DimNormal,
                                               DimNormal>(new FunctorSmoothnessNormal());
    }
};
}
