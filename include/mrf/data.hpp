#pragma once

#include <Eigen/Eigen>
#include <opencv/cxcore.h>
#include <memory>

namespace mrf{

class Data {
public:
	using Ptr = std::shared_ptr<Data>;
public:
    Data(const cv::Mat& cv_image, const Eigen::VectorXf& depth, const Eigen::VectorXf& certainty, const int width);
    Data(const  Eigen::VectorXf& eigen_image, const Eigen::VectorXf& depth, const Eigen::VectorXf& certainty,const int width);
    Data(const int dim);
    Data(){};

    static Data::Ptr create(const cv::Mat& cv_image, const Eigen::VectorXf& depth, const Eigen::VectorXf& certainty,const int width);
    static Data::Ptr create(const  Eigen::VectorXf& eigen_image, const Eigen::VectorXf& depth,const Eigen::VectorXf& certainty, const int width);
    Data& operator=(const Data& in);

     Eigen::VectorXf image;
     Eigen::VectorXf depth;
     Eigen::VectorXf certainty;

     int width;
private:
     friend std::ostream& operator<<(std::ostream& os, const Data& f);

};
}//end of mrf namespace
