#pragma once

#include <memory>
#include <camera_models/camera_model.h>

#include "data.hpp"
#include "parameters.hpp"

namespace mrf {

class Solver {

public:
    using Ptr = std::shared_ptr<Solver>;

    inline Solver(const std::shared_ptr<CameraModel>& cam, const Parameters& p = Parameters())
            : camera_(cam), params_(p){};

    template <typename T>
    bool solve(const Data<T>&, Data<T>&, const bool pin_transform = true);

    inline static Ptr create(const std::shared_ptr<CameraModel>& cam,
                             const Parameters& p = Parameters()) {
        return std::make_shared<Solver>(cam, p);
    }

private:
    void getNNdepths(Eigen::VectorXd& depth_est);
    const std::shared_ptr<CameraModel> camera_;
    Parameters params_;
};
}

#include "internal/solver_impl.hpp"
