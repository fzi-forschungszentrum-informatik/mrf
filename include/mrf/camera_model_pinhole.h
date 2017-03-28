#ifndef CAMERAMODELPINHOLE_H
#define CAMERAMODELPINHOLE_H
#include <array>
#include <memory>

#include "camera_model.h"
#include "camera_model_pinhole_functions.h"

class CameraModelPinhole : public CameraModel {

public:
    CameraModelPinhole();
    virtual ~CameraModelPinhole();

    CameraModelPinhole(double focalLength);
    CameraModelPinhole(int imgWidth,
                       int imgHeight,
                       double focalLength,
                       double principalPointU,
                       double principalPointV);

    void init(int imgWidth,
              int imgHeight,
              double focalLength,
              double principalPointU,
              double principalPointV);

    virtual void getImageSize(int& imgWidth, int& imgHeight) const override;

    virtual bool getImagePoint(const Eigen::Ref<const Eigen::Vector3d>& point3d,
                               Eigen::Ref<Eigen::Vector2d> imagePoint) const override;
    virtual bool getViewingRay(const Eigen::Ref<const Eigen::Vector2d>& imagePoint,
                               Eigen::Ref<Eigen::Vector3d> supportPoint,
                               Eigen::Ref<Eigen::Vector3d> direction) const override;

private:
    int imgWidth_;
    int imgHeight_;
    Eigen::Matrix<double, 3, 1> intrinsics_;
};


#endif // CAMERAMODELPINHOLE_H
