#pragma once

#include <Eigen/Eigen>
#include "mrf_data.hpp"

namespace mrf {

class MrfDataEigenCg : public MrfData {
public:
    MrfDataEigenCg(Eigen::VectorXf& image_in, Eigen::VectorXf& z_in, std::vector<int> indices_in,
                   int width_in, int ks_in, int kd_in);
    MrfDataEigenCg(Eigen::VectorXf& image_in, Eigen::Matrix3Xf& points_in,
                   std::vector<int> indices_in, int width_in, int ks_in, int kd_in);

    Eigen::VectorXf image;
    Eigen::VectorXf z;
    std::vector<int> indices;
    int width;
    int ks;
    int kd;

    MrfDataEigenCg(){};
    MrfDataEigenCg(MrfDataEigenCg&& c)
            : image(std::move(c.image)), z(std::move(c.z)), indices(std::move(c.indices)),
              width(std::move(c.width)), ks(std::move(c.ks)), kd(std::move(c.kd)) {
    }
    MrfDataEigenCg& operator=(MrfDataEigenCg&& c) {
        if (&c != this) {
            image = std::move(c.image);
            z = std::move(c.z);
            indices = std::move(c.indices);
            width = std::move(c.width);
            ks = std::move(c.ks);
            kd = std::move(kd);
        }
        return *this;
    }

    MrfDataEigenCg& operator=(MrfDataEigenCg& c) {
        if (&c != this) {
            image = c.image;
            z = c.z;
            indices = c.indices;
            width = c.width;
            ks = c.ks;
            kd = c.kd;
        }
        return *this;
    }

    MrfDataEigenCg clone();

private:
};
}
