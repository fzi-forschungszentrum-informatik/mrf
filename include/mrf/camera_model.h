#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Eigen>

/** @brief Types of camera models specified */
enum class CameraModelType {
    CAMERA_MODEL_NON_SVP,
    CAMERA_MODEL_NURBS,
    CAMERA_MODEL_PINHOLE,
    CAMERA_MODEL_SPHERE,
    CAMERA_MODEL_SVP,
    CAMERA_MODEL_EPIPOLAR,
    CAMERA_MODEL_NON_SVP_GENERIC_DISTORTION,
    CAMERA_MODEL_FAST_LOOKUP,

    // this must be the last entry, so do not edit this
    CAMERA_MODEL_NUM
};

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

    virtual bool isSvp() const = 0;
    virtual double getFocalLength() const = 0;
    virtual Eigen::Vector2d getPrincipalPoint() const = 0;

    virtual std::unique_ptr<CameraModel> clone() const = 0;

    //    void load(cereal::PortableBinaryInputArchive& archive);
    //    void save(cereal::PortableBinaryOutputArchive& archive) const;

    virtual CameraModelType getId() const = 0;

    const std::string& getName() const;
    void setName(const std::string& name);

protected:
    CameraModel() = default;

private:
    std::string name_;
};

std::ostream& operator<<(std::ostream& out, CameraModel& cam);
