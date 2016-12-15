#pragma once

#include <memory>
#include <camera_models/camera_model.h>

#include "data.hpp"
#include "parameters.hpp"

namespace mrf {

class Solver {

    enum class NeighbourCase { top_bottom, left_right, top_left_right, bottom_left_right };

public:
    using Ptr = std::shared_ptr<Solver>;

    inline Solver(const std::unique_ptr<CameraModel>& cam, const Parameters& p = Parameters())
            : camera_{std::move(cam)}, params_(p){};

    template <typename T>
    bool solve(Data<T>&);

    inline static Ptr create(std::unique_ptr<CameraModel> cam, const Parameters& p = Parameters()) {
        return std::make_shared<Solver>(cam, p);
    }

private:
//    double neighbourDiff(const int p, const int pnext, const NeighbourCase& nc, const int width,
//                        const int dim);
//    double diff(const double depth_i, const double depth_j);
    std::vector<double> smoothnessWeights(const int p,const std::vector<int>&neighbours,const cv::Mat& img);
    bool solveCeres(Data&);

    std::unique_ptr<CameraModel> camera_;
    Parameters params_;
};
}

#include "internal/solver_impl.hpp"
