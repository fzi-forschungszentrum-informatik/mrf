#pragma once

#include <memory>
#include <camera_models/camera_model.h>
#include <pcl/point_types.h>

#include "data.hpp"
#include "parameters.hpp"
#include "result_info.hpp"

namespace mrf {

class Solver {

public:
    using Ptr = std::shared_ptr<Solver>;
    using PointT = pcl::PointXYZINormal;

    inline Solver(const std::shared_ptr<CameraModel>& cam, const Parameters& p = Parameters())
            : camera_(cam), params_(p){};

    template <typename T>
    ResultInfo solve(const Data<T>&, Data<PointT>&, const bool pin_transform = true);

    inline Data<PointT> getDebugInfo() const {
        return d_;
    }

    inline static Ptr create(const std::shared_ptr<CameraModel>& cam,
                             const Parameters& p = Parameters()) {
        return std::make_shared<Solver>(cam, p);
    }

private:
    void getNNdepths(Eigen::VectorXd& depth_est);
    const std::shared_ptr<CameraModel> camera_;
    Parameters params_;
    Data<PointT> d_;
};
}

#include "internal/solver_impl.hpp"
