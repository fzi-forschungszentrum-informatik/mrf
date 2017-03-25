#include <Eigen/Geometry>

#include "camera_model_pinhole.h"
#include "eigen.hpp"

CameraModelPinhole::CameraModelPinhole(void) {
}

CameraModelPinhole::~CameraModelPinhole(void) {
}

CameraModelPinhole::CameraModelPinhole(double focalLength) {
    init(-1, -1, focalLength, 0.0, 0.0);
}

CameraModelPinhole::CameraModelPinhole(int imgWidth,
                                       int imgHeight,
                                       double focalLength,
                                       double principalPointU,
                                       double principalPointV) {
    init(imgWidth, imgHeight, focalLength, principalPointU, principalPointV);
}

void CameraModelPinhole::init(int imgWidth,
                              int imgHeight,
                              double focalLength,
                              double principalPointU,
                              double principalPointV) {
    imgWidth_ = imgWidth;
    imgHeight_ = imgHeight;
    intrinsics_[0] = focalLength;
    intrinsics_[1] = principalPointU;
    intrinsics_[2] = principalPointV;
}

void CameraModelPinhole::getImageSize(int& imgWidth, int& imgHeight) const {
    imgWidth = imgWidth_;
    imgHeight = imgHeight_;
}

bool CameraModelPinhole::getImagePoint(const Eigen::Ref<const Eigen::Vector3d>& point3d,
                                       Eigen::Ref<Eigen::Vector2d> imagePoint) const {
    CameraModelPinholeGetImagePoint getImagePoint;
    return getImagePoint(double(), intrinsics_, point3d, imagePoint);
}

bool CameraModelPinhole::getViewingRay(const Eigen::Ref<const Eigen::Vector2d>& imagePoint,
                                       Eigen::Ref<Eigen::Vector3d> supportPoint,
                                       Eigen::Ref<Eigen::Vector3d> direction) const {
    CameraModelPinholeGetViewingRay getViewingRay;
    return getViewingRay(double(), intrinsics_, imagePoint, supportPoint, direction);
}

bool CameraModelPinhole::isSvp() const {
    return true;
}

double CameraModelPinhole::getFocalLength() const {
    return intrinsics_[0];
}

Eigen::Vector2d CameraModelPinhole::getPrincipalPoint() const {
    return Eigen::Vector2d(intrinsics_[1], intrinsics_[2]);
}

std::unique_ptr<CameraModel> CameraModelPinhole::clone() const {
    return std::unique_ptr<CameraModelPinhole>(
        new CameraModelPinhole(*this)); // copy construct new element
}

CameraModelType CameraModelPinhole::getId() const {
    return ID;
}
