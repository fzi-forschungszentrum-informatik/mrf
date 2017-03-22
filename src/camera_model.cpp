#include "camera_model.h"

/** @brief Determines which 3D points are in front of the image.
 *  @param point3d 3D points to check
 *  @param imagePoint Image points. Will have the according 3D point value if it's in front
 *  @return Vector representing which 3D points are in front of the image */
std::vector<bool> CameraModel::getImagePoints(const Eigen::Ref<const Eigen::Matrix3Xd>& point3d,
                                              Eigen::Ref<Eigen::Matrix2Xd> imagePoint) const {

    const int N{(int)point3d.cols()};

    if (imagePoint.cols() != N) {
        throw std::runtime_error("Output matrix imagePoint has wrong size!");
    }

    std::vector<bool> return_val(N);
    for (int j = 0; j < N; j++) {
        return_val[j] = getImagePoint(point3d.template block<3, 1>(0, j),
                                      imagePoint.template block<2, 1>(0, j));
    }

    return return_val;
}

/** @brief Get viewing rays from the camera origin to all pixel
 *  @param imagePoint Image points
 *  @param supportPoint Support point of the viewing rays
 *  @param direction Direction of the viewing rays
 *  @return Vector representing if a viewing ray could be calculated for that pixel */
std::vector<bool> CameraModel::getViewingRays(const Eigen::Ref<const Eigen::Matrix2Xd>& imagePoint,
                                              Eigen::Ref<Eigen::Matrix3Xd> supportPoint,
                                              Eigen::Ref<Eigen::Matrix3Xd> direction) const {

    const int N{(int)imagePoint.cols()};

    if (supportPoint.cols() != N) {
        throw std::runtime_error("Output matrix supportPoint has wrong size!");
    }
    if (direction.cols() != N) {
        throw std::runtime_error("Output matrix direction has wrong size!");
    }

    std::vector<bool> return_val(N);
    for (int j = 0; j < N; j++) {
        return_val[j] = getViewingRay(imagePoint.template block<2, 1>(0, j),
                                      supportPoint.template block<3, 1>(0, j),
                                      direction.template block<3, 1>(0, j));
    }

    return return_val;
}

/** @brief Get the name of the camera
 *  @return Name of the camera */
const std::string& CameraModel::getName() const {
    if (name_.empty() == true) {
        throw std::runtime_error(
            "No camera name specified! Call setName in derived camera model class.");
    }
    return name_;
}

/** @brief Set the name of the camera.
 *  @attention Can only be done once!
 *  @param name Name of the camera */
void CameraModel::setName(const std::string& name) {
    if (name_.empty() == false) {
        throw std::runtime_error("Camera name was already specified!");
    }
    name_ = name;
}

std::ostream& operator<<(std::ostream& out, CameraModel& cam) {
    out << "camera " << cam.getName();

    int imgWidth, imgHeight;
    cam.getImageSize(imgWidth, imgHeight);
    out << " [" << imgWidth << " x " << imgHeight << "]" << std::endl;

    out << " - f = " << cam.getFocalLength() << std::endl;
    out << " - cu = " << cam.getPrincipalPoint().x() << std::endl;
    out << " - cv = " << cam.getPrincipalPoint().y() << std::endl;

    return out;
}
