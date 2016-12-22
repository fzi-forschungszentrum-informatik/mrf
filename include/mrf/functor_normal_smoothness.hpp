#pragma once

#include <ostream>
#include <ceres/autodiff_cost_function.h>

namespace mrf {

struct FunctorNormalSmoothness {

    static constexpr size_t DimNormal = 1;
    static constexpr size_t DimResidual = 3;

    inline FunctorNormalSmoothness(const double& w) : w_{w} {};

    template <typename T>
    inline bool operator()(const T* const nix, const T* const niy, const T* const niz,
                           const T* const njx, const T* const njy, const T* const njz,
                           T* res) const {
        res[0] = T(w_) * (nix[0] - njx[0]);
        res[1] = T(w_) * (niy[1] - njy[2]);
        res[2] = T(w_) * (niz[1] - njz[2]);
        return true;
    }

    inline static ceres::CostFunction* create(const double& e) {
        return new ceres::AutoDiffCostFunction<FunctorNormalSmoothness, DimResidual, DimNormal,
                                               DimNormal, DimNormal, DimNormal, DimNormal,
                                               DimNormal>(new FunctorNormalSmoothness(e));
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorNormalSmoothness& f) {
        return os << "Weight: " << f.w_ << std::endl;
    }

private:
    const double w_;
};
}
