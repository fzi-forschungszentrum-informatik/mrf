#pragma once

#include <memory>
#include <ostream>
#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorNormal {

    using Ptr = std::shared_ptr<FunctorNormal>;

    static constexpr size_t DimNormal = 3;
    static constexpr size_t DimResidual = 3;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormal(const Eigen::Vector3d& n, const double& w) : n_{n}, w_{w} {};

    template <typename T>
    inline bool operator()(const T* const normal, const T* const rotation, T* res) const {
        using namespace Eigen;
        const Map<const Vector3<T>> n(normal);
        Map<Vector3<T>>(res, DimResidual) =
            T(w_) * (n - util_ceres::fromQuaternion(rotation) * n_.cast<T>());
        return true;
    }

    inline static Ptr create(const Eigen::Vector3d& n, const double& w) {
        return std::make_shared<FunctorNormal>(n, w);
    }

    inline ceres::CostFunction* toCeres() {
        return new ceres::AutoDiffCostFunction<FunctorNormal, DimResidual, DimNormal, DimRotation>(
            this);
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
