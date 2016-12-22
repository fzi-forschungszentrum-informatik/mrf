#pragma once

#include <memory>
#include <ostream>
#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorNormal {

    using Ptr = std::shared_ptr<FunctorNormal>;

    static constexpr size_t DimNormal = 1;
    static constexpr size_t DimResidual = 3;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormal(const Eigen::Vector3d& n, const double& w) : n_{n}, w_{w} {};

    template <typename T>
    inline bool operator()(const T* const normal_x, const T* const normal_y,
                           const T* const normal_z, const T* const rotation, T* res) const {
        using namespace Eigen;
        const Affine3<T> tf{util_ceres::fromQuaternion(rotation)};
        Map<Vector3<T>>(res, DimResidual) =
            T(w_) *
            (Eigen::Vector3<T>(normal_x[0], normal_y[0], normal_z[0]) - tf * n_.cast<T>());
        return true;
    }

    inline static Ptr create(const Eigen::Vector3d& n, const double& w) {
        return std::make_shared<FunctorNormal>(n, w);
    }

    inline ceres::CostFunction* toCeres() {
        return new ceres::AutoDiffCostFunction<FunctorNormal, DimResidual, DimNormal, DimNormal,
                                               DimNormal, DimRotation>(this);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorNormal& f) {
        return os << "Vector: " << f.n_ << std::endl << "Weight: " << f.w_ << std::endl;
    }

    inline void setNormal(const Eigen::Vector3d& p) {
        n_ = p;
    }
    inline void setWeight(const double w) {
        w_ = w;
    }

private:
    Eigen::Vector3d n_; ///< 3D world point normal associated to this pixel
    double w_;          ///< Weight
};
}
