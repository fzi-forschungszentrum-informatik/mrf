#include <mrf_data_eigen_cg.hpp>

namespace mrf {

MrfDataEigenCg::MrfDataEigenCg(Eigen::VectorXf& image_in, Eigen::VectorXf& z_in,
                               std::vector<int> indices_in, int width_in, int ks_in, int kd_in){

    if(!(ks_in == 0 && kd_in > 0))  throw std::runtime_error("Invalid ks kd inputs ");
    if(!(width_in > 0 && width_in < image_in.cols()))  throw std::runtime_error("Invalid MrfData width/image input ");
    if(!(indices_in.size() > 0))  throw std::runtime_error("Invalid MrfData indices");
    if(!(image_in.cols() > indices_in.size()))  throw std::runtime_error("Invalid MrfData image/indices input");

    if (image_in.cols() == z_in.cols() && indices_in.size() == z_in.nonZeros()) {
        image = image_in;
        z = z_in;
    } else if (indices_in.size() == z_in.cols()) {
    	image = image_in;
        z.setZero(image_in.cols());
        for (int i = 0; i < indices_in.size(); i++) {
            z(i) = z_in(i);
        }
    }
    else {
    	throw std::runtime_error("Invalid MrfDataEigenCg Inputs ");
    }

    indices = indices_in;
    width = width_in;
    ks = ks_in;
    kd = kd_in;
}

MrfDataEigenCg::MrfDataEigenCg(Eigen::VectorXf& image_in, Eigen::Matrix3Xf& points_in,
                               std::vector<int> indices_in, int width_in, int ks_in, int kd_in){
    assert(ks_in > 0 && kd_in > 0);
    assert(width_in > 0 && width_in < image_in.cols());
    assert(indices_in.size() > 0);
    assert(image_in.cols() > indices_in.size());

    z.setZero(image_in.cols());
    for (int i = 0; i < indices_in.size(); i++) {
        z(i) = points_in.col(i).norm();
    }
    indices = indices_in;
        width = width_in;
        ks = ks_in;
        kd = kd_in;
}

MrfDataEigenCg MrfDataEigenCg::clone() {
    MrfDataEigenCg out;
    out.image = this->image;
    out.indices = this->indices;
    out.z = this->z;
    out.width = this->width;
    out.ks = this->ks;
    out.kd = this->kd;
    return out;
}
}
