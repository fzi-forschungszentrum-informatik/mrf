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

    inline FunctorNormal(const Eigen::Vector3d& n) : n_{n} {};

    template <typename T>
    inline bool operator()(const T* const normal, const T* const rotation, T* res) const {
        using namespace Eigen;
        Map<Vector3<T>>(res, DimResidual) =
            Map<const Vector3<T>>(normal) - util_ceres::fromQuaternion(rotation) * n_.cast<T>();
        return true;
    }

    inline static Ptr create(const Eigen::Vector3d& n) {
        return std::make_shared<FunctorNormal>(n);
    }

    inline ceres::CostFunction* toCeres() {
        return new ceres::AutoDiffCostFunction<FunctorNormal, DimResidual, DimNormal, DimRotation>(
            this);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorNormal& f) {
        return os << "Vector: " << f.n_ << std::endl;
    }

    inline void setNormal(const Eigen::Vector3d& p) {
        n_ = p;
    }

private:
    Eigen::Vector3d n_; ///< 3D world point normal associated to this pixel
};
}
