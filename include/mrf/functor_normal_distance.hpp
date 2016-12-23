#pragma once

#include <memory>
#include <ostream>
#include <ceres/autodiff_cost_function.h>
#include <Eigen/src/Geometry/Hyperplane.h>
#include <Eigen/src/Geometry/ParametrizedLine.h>

namespace mrf {

struct FunctorNormalDistance {

    using Ptr = std::shared_ptr<FunctorNormalDistance>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 3;
    static constexpr size_t DimNormal = 3;

    inline FunctorNormalDistance(const double& w,
                                 const Eigen::ParametrizedLine<double, 3>& ray_this,
                                 const Eigen::ParametrizedLine<double, 3>& ray_nn)
            : w_{w}, ray_this_{ray_this}, ray_nn_{ray_nn} {};

    template <typename T>
    inline bool operator()(const T* const depth_this, const T* const depth_nn,
                           const T* const normal_this, T* res) const {
        using namespace Eigen;
        const Map<const Vector3<T>> n(normal_this);
        const Hyperplane<T, 3> plane_this(n, ray_this_.cast<T>().pointAt(depth_this[0]));

        //        Map<Vector3<T>>(res, DimResidual) =
        //            T(w_) * (ray_nn_.cast<T>().intersectionPoint(plane_this) -
        //                     ray_nn_.cast<T>().pointAt(depth_nn[0]));

        const Vector3<T> p_nn{ray_nn_.cast<T>().pointAt(depth_nn[0])};
        Map<Vector3<T>>(res, DimResidual) = T(w_) * (plane_this.projection(p_nn) - p_nn);

        return true;
    }

    inline static ceres::CostFunction* create(const double& w,
                                              const Eigen::ParametrizedLine<double, 3>& ray_this,
                                              const Eigen::ParametrizedLine<double, 3>& ray_nn) {
        return new ceres::AutoDiffCostFunction<FunctorNormalDistance, DimResidual, DimDepth,
                                               DimDepth, DimNormal>(
            new FunctorNormalDistance(w, ray_this, ray_nn));
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorNormalDistance& f) {
        return os << "Weight: " << f.w_ << std::endl;
    }

    inline void setWeight(const double w) {
        w_ = w;
    }

private:
    double w_; ///< Weight
    const Eigen::ParametrizedLine<double, 3> ray_this_, ray_nn_;
};
}
