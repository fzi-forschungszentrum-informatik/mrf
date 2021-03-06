#pragma once

#include <Eigen/Geometry>
#include <ceres/autodiff_cost_function.h>
#include "eigen.hpp"
namespace mrf {

struct FunctorCollinearity {

    template <typename T>
    using Ray = Eigen::ParametrizedLine<T, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 3;

    inline FunctorCollinearity(const Ray<double>& ray_0,
                               const Ray<double>& ray_1,
                               const Ray<double>& ray_2)
            : ray_0_{ray_0}, ray_1_{ray_1}, ray_2_{ray_2} {};

    template <typename T>
    inline bool operator()(const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           T* res) const {
        using namespace Eigen;
        const Vector3<T> p_0{ray_0_.cast<T>().pointAt(d_0[0])};
        const Vector3<T> p_01{p_0 - ray_1_.cast<T>().pointAt(d_1[0])};
        const Vector3<T> p_20{ray_2_.cast<T>().pointAt(d_2[0]) - p_0};
        Map<Vector3<T>>(res, DimResidual) =
             p_20.norm() * p_01 - p_01.norm() * p_20;
        return true;
    }

    inline static ceres::CostFunction* create(const Ray<double>& ray_0,
                                              const Ray<double>& ray_1,
                                              const Ray<double>& ray_2) {
        return new ceres::
            AutoDiffCostFunction<FunctorCollinearity, DimResidual, DimDepth, DimDepth, DimDepth>(
                new FunctorCollinearity(ray_0, ray_1, ray_2));
    }

private:
    const Ray<double> ray_0_;
    const Ray<double> ray_1_;
    const Ray<double> ray_2_;
};
}
