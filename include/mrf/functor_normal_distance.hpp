#pragma once

#include <Eigen/Geometry>
#include <ceres/autodiff_cost_function.h>

namespace mrf {

struct FunctorNormalDistance {

    template <typename T>
    using Ray = Eigen::ParametrizedLine<T, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 3;

    inline FunctorNormalDistance(const Ray<double>& ray_0,
                                 const Ray<double>& ray_1,
                                 const Ray<double>& ray_2,
                                 const double& w)
            : ray_0_{ray_0}, ray_1_{ray_1}, ray_2_{ray_2}, w_{w} {};

    template <typename T>
    inline bool operator()(const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           T* res) const {
        using namespace Eigen;

        const Vector3<T> p_0{ray_0_.cast<T>().pointAt(d_0[0])};
        const Vector3<T> p_1{ray_1_.cast<T>().pointAt(d_1[0])};
        const Vector3<T> p_2{ray_2_.cast<T>().pointAt(d_2[0])};
        Map<Vector3<T>>(res, DimResidual) = static_cast<T>(w_) * (-p_1 + T(2) * p_0 - p_2);

        //
        //        const Vector3<T> d_01{(p_0 - p_1).normalized()};
        //        const Vector3<T> d_20{(p_2 - p_0).normalized()};
        //
        //
        //        if (norm_01 < T(1.e-13) || norm_20 < T(1.e-13))
        //            std::cerr << "ERROR!!!!";
        //        Vector3<T> res2{static_cast<T>(w_) * (d_01 / norm_01 - d_20 / norm_20)};
        //        std::cout << "res2: " << res2[0] << "\n" << res2[1] << "\n" << res2[2] <<
        //        std::endl;
        //        Map<Vector3<T>>(res, DimResidual) = static_cast<T>(w_) * (d_01 - d_20);
        //        res[0] = T(w_) * (T(1) - d_01.dot(d_20));
        return true;
    }

    inline static ceres::CostFunction* create(const Ray<double>& ray_0,
                                              const Ray<double>& ray_1,
                                              const Ray<double>& ray_2,
                                              const double& w) {
        return new ceres::
            AutoDiffCostFunction<FunctorNormalDistance, DimResidual, DimDepth, DimDepth, DimDepth>(
                new FunctorNormalDistance(ray_0, ray_1, ray_2, w));
    }

private:
    const Ray<double> ray_0_;
    const Ray<double> ray_1_;
    const Ray<double> ray_2_;
    const double w_;
};
}
