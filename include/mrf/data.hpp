#pragma once

#include <memory>
#include <pcl/point_cloud.h>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

namespace mrf {

/** @brief Storage for data used in the optimization problem.
    Stores the pointcloud, the image and the transformation.  */
template <typename T>
struct Data {
    using Ptr = std::shared_ptr<Data>;

    using Cloud = pcl::PointCloud<T>;
    using Image = cv::Mat;
    using Transform = Eigen::Affine3d;

    inline Data() : cloud{new Cloud}, transform{Transform::Identity()} {};
    inline Data(const typename Cloud::Ptr& cl,
                const Image& img,
                const Transform& tf = Transform::Identity())
            : cloud{cl}, image{img}, transform{tf} {};

    inline static Ptr create(const typename Cloud::Ptr& cl,
                             const Image& img,
                             const Transform& tf = Transform::Identity()) {
        return std::make_shared<Data>(cl, img, tf);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Data& d) {
        os << "Image size: " << d.image.cols << " x " << d.image.rows << std::endl
           << "Number of cloud points: " << d.cloud->size() << std::endl
           << "Transform: \n"
           << d.transform.matrix();
        return os;
    }

    typename Cloud::Ptr cloud;      ///< Original cloud
    Image image;                    ///< Camera image
    Transform transform;            ///< Transform between camera and laser
};
}
