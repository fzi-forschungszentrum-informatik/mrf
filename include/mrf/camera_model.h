#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Eigen>

class CameraModel {
public:
    virtual ~CameraModel() {
    }

    virtual void getImageSize(int& imgWidth, int& imgHeight) const = 0;

    virtual bool getImagePoint(const Eigen::Ref<const Eigen::Vector3d>& point3d,
                               Eigen::Ref<Eigen::Vector2d> imagePoint) const = 0;
    virtual bool getViewingRay(const Eigen::Ref<const Eigen::Vector2d>& imagePoint,
                               Eigen::Ref<Eigen::Vector3d> supportPoint,
                               Eigen::Ref<Eigen::Vector3d> direction) const = 0;

    virtual std::vector<bool> getImagePoints(const Eigen::Ref<const Eigen::Matrix3Xd>& point3d,
                                             Eigen::Ref<Eigen::Matrix2Xd> imagePoint) const;
    virtual std::vector<bool> getViewingRays(const Eigen::Ref<const Eigen::Matrix2Xd>& imagePoint,
                                             Eigen::Ref<Eigen::Matrix3Xd> supportPoint,
                                             Eigen::Ref<Eigen::Matrix3Xd> direction) const;
};

std::ostream& operator<<(std::ostream& out, CameraModel& cam);
