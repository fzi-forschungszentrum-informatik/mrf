#pragma once

#include <ostream>
#include <ceres/autodiff_cost_function.h>

namespace mrf {

struct FunctorSmoothnessNormal {

    static constexpr size_t DimNormal = 3;
    static constexpr size_t DimResidual = 3;

    inline FunctorSmoothnessNormal(const double& w) : w_{w} {};

    template <typename T>
    inline bool operator()(const T* const n_this, const T* const n_nn, T* res) const {
        res[0] = T(w_) * (n_this[0] - n_nn[0]);
        res[1] = T(w_) * (n_this[1] - n_nn[2]);
        res[2] = T(w_) * (n_this[1] - n_nn[2]);
        return true;
    }

    inline static ceres::CostFunction* create(const double& e) {
        return new ceres::AutoDiffCostFunction<FunctorSmoothnessNormal, DimResidual, DimNormal,
                                               DimNormal>(new FunctorSmoothnessNormal(e));
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorSmoothnessNormal& f) {
        return os << "Weight: " << f.w_ << std::endl;
    }

private:
    const double w_;
};
}
