
#include <data.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "generic_logger/generic_logger.hpp"

namespace mrf {
Data::Data(const cv::Mat& cv_image, const Eigen::VectorXf& depth, const Eigen::VectorXf& certainty,
           const int width) {
    if (!((cv_image.rows * cv_image.cols == depth.size()) && (width > 0) &&
          (width < depth.size()) && (cv_image.cols == width))) {
        ERROR_STREAM("cv size: " << cv_image.cols * cv_image.rows);
        ERROR_STREAM("depth size: " << depth.size());
        ERROR_STREAM("certainty size: " << certainty.size());
        ERROR_STREAM("width: " << width);
        throw std::runtime_error("Input dimension do not match");
    }
    if (depth != depth) {
            ERROR_STREAM("Data constructor: depth contains nan");
   }


    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
    if (cv_image.type() != 0) {
        cv::Mat temp;
        ;
        cv_image.convertTo(temp, 0);
        cv::cv2eigen(temp, eigen_mat);
    } else {
        cv::cv2eigen(cv_image, eigen_mat);
    }
    if (eigen_mat.isZero()) {
        throw std::runtime_error("Cv->Eigen conversion failed");
    }
    if (!(eigen_mat.size() == depth.size() && certainty.size() == depth.size())) {
        throw std::runtime_error(" image and depth dimensions do not match");
    }
    this->image = eigen_mat;
    if (image.size() != depth.size()) {
        throw std::runtime_error(" image and depth dimensions do not match");
    }
    this->certainty = certainty;
    this->depth = depth;
    this->width = width;
}

Data::Data(const Eigen::VectorXf& eigen_image, const Eigen::VectorXf& depth,
           const Eigen::VectorXf& certainty, const int width) {
    if (!(eigen_image.size() == depth.size() && width > 0 && width < depth.size() &&
          depth.size() == certainty.size())) {
        ERROR_STREAM("eigen_image size: " << eigen_image.size());
        ERROR_STREAM("depth size: " << depth.size());
        ERROR_STREAM("certainty size: " << certainty.size());
        ERROR_STREAM("width: " << width);
        throw std::runtime_error("Input dimension do not match");
    }
    image = eigen_image;
    this->certainty = certainty;
    this->depth = depth;
    this->width = width;
}

Data::Data(const int dim) {
    image.setZero(dim);
    depth.setZero(dim);
    certainty.setZero(dim);
}

Data& Data::operator=(const Data& in) {
    if (in.depth != in.depth) {
        ERROR_STREAM("in.depth contains nan");
    }
    if (in.image != in.image) {
        ERROR_STREAM("in.image contains nan");
    }
    if (in.certainty != in.certainty) {
        ERROR_STREAM("in.certainty contains nan");
    }
    this->image = in.image;
    this->depth = in.depth;
    this->certainty = in.certainty;
    this->width = in.width;
    return *this;
}

Data::Ptr Data::create(const cv::Mat& cv_image, const Eigen::VectorXf& depth,
                       const Eigen::VectorXf& certainty, const int width) {
    return std::make_shared<Data>(cv_image, depth, certainty, width);
}

Data::Ptr Data::create(const Eigen::VectorXf& eigen_image, const Eigen::VectorXf& depth,
                       const Eigen::VectorXf& certainty, const int width) {
    return std::make_shared<Data>(eigen_image, depth, certainty, width);
}

std::ostream& operator<<(std::ostream& os, const Data& f) {
    os << "Data: " << std::endl;
    os << "image size: " << f.image.size() << std::endl;
    os << "image max,min: " << f.image.maxCoeff() << ", " << f.image.minCoeff() << std::endl;
    os << "depth size: " << f.depth.size() << std::endl;
    os << "depth max,min: " << f.depth.maxCoeff() << ", " << f.depth.minCoeff() << std::endl;
    os << "certainty size: " << f.certainty.size() << std::endl;
    os << "certainty max,min: " << f.certainty.maxCoeff() << ", " << f.certainty.minCoeff()
       << std::endl;
    os << "width: " << f.width << std::endl;
}

} // end of mrf namespace
