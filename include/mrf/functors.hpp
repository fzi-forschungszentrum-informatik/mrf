#pragma once

#include <Eigen/Eigen>
#include <ceres/ceres.h>

namespace mrf {

class CostFunctor {

    template <typename T>
    using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    static constexpr size_t DimParam = 1;
    static constexpr size_t DimRes = 1;

public:
    CostFunctor(const double& zi, const double& kd) : zi_{zi}, kd_{kd} {};

    template <typename T>
    inline bool operator()(const T* const x, T* residual) const{

    }

    inline static ceres::CostFunction* create(const double& z, const double& kd) {
        return new ceres::AutoDiffCostFunction<CostFunctor, CostFunctor::DimRes,
                                               CostFunctor::DimParam>(
            new CostFunctor(z, kd));
    }

private:
    const double zi_;
    const double kd_;
    };

class SmoothFunctor {
public:
    static constexpr size_t DimParam = 1;
    static constexpr size_t DimRes = 1;

public:
    inline SmoothFunctor(const double& sqrt_e) : sqrt_e_{sqrt_e} {
    }

    template <typename T>
    inline bool operator()(const T* const xi, const T* const xj, T* res) const {
        res[0] = T(sqrt_e_) * (xi[0] - xj[0]);
        return true;
    }

    inline static ceres::CostFunction* create(const double& e) {
        return new ceres::AutoDiffCostFunction<SmoothFunctor, DimRes, DimParam, DimParam>(
            new SmoothFunctor(e));
    }

private:
    const double sqrt_e_;

};

} // end of mrf namespace
