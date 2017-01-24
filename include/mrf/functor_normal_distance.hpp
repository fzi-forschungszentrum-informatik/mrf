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

        const Vector3<T> p_01{p_0 - p_1};
        const Vector3<T> p_20{p_2 - p_0};

        /**
         * Secant rate on projections (problems on edges)
         */
        //        const Vector3<T> projection_1{ray_0_.cast<T>().projection(p_1)};
        //        const Vector3<T> projection_2{ray_0_.cast<T>().projection(p_2)};
        //        const T y_1{ray_0_.cast<T>().distance(p_1)};
        //        const T y_2{ray_0_.cast<T>().distance(p_2)};
        //        const T x_1{(p_0 - projection_1).norm()};
        //        const T x_2{(projection_2 - p_0).norm()};
        //        res[0] = x_2 * y_1 - x_1 * y_2;

        /**
         *
         */
        //        const Vector3<T> projection_1{ray_0_.cast<T>().projection(p_1)};
        //        const Vector3<T> projection_2{ray_0_.cast<T>().projection(p_2)};
        //        Map<Vector3<T>>(res, DimResidual) =
        //            projection_1 - p_1 + p_0 - projection_1 - (projection_2 - p_2 + p_0 -
        //            projection_2);

        /**
         * 2nd difference (best)
         */
        Map<Vector3<T>>(res, DimResidual) = static_cast<T>(w_) * (p_01 - p_20);

        /**
         * Dot product of difference vectors
         */
        //        res[0] = static_cast<T>(w_) * (p_01.norm() * p_20.norm() - p_01.dot(p_20));

        /**
         * Difference of normals
         */
        //        using Plane = Hyperplane<T, 3>;
        //        const Plane plane_1{Plane::Through(p_0, p_1, ray_0_.cast<T>().projection(p_1))};
        //        const Plane plane_2{Plane::Through(p_0, p_2, ray_0_.cast<T>().projection(p_2))};
        //        Map<Vector3<T>>(res, DimResidual) =
        //            static_cast<T>(w_) * (plane_1.normal() - plane_2.normal());

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
