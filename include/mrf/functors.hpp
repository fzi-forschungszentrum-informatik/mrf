#pragma once

#include <ceres/autodiff_cost_function.h>

namespace mrf {

class CostFunctor {

    static constexpr size_t DimParam = 1;
    static constexpr size_t DimRes = 1;

public:
    inline CostFunctor(const double& z, const double& w) : z_{z}, w_{w} {};

    template <typename T>
    inline bool operator()(const T* const x, T* res) const {
        res[0] = T(w_) * (x[0] - T(z_));
        return true;
    }

    inline static ceres::CostFunction* create(const double& z, const double& kd) {
        return new ceres::AutoDiffCostFunction<CostFunctor, DimRes, DimParam>(
            new CostFunctor(z, kd));
    }

private:
    const double z_;
    const double w_;
};

class SmoothFunctor {

    static constexpr size_t DimParam = 1;
    static constexpr size_t DimRes = 1;

public:
    inline SmoothFunctor(const double& w) : w_{w} {
    }

    template <typename T>
    inline bool operator()(const T* const xi, const T* const xj, T* res) const {
        res[0] = T(w_) * (xi[0] - xj[0]);
        return true;
    }

    inline static ceres::CostFunction* create(const double& e) {
        return new ceres::AutoDiffCostFunction<SmoothFunctor, DimRes, DimParam, DimParam>(
            new SmoothFunctor(e));
    }

private:
    const double w_;
};

//class LossFunctor{
//public:
//	inline LossFunctor(const double& w) : w_{w}{};
//	inline static ceres::LossFunction* create(const double&w){
//		return new ceres::ScaledLoss(new LossFunctor(w),w,ceres::Ownership::TAKE_OWNERSHIP);
//	}
//
//private:
//	const double w_;
//};

} // end of mrf namespace
