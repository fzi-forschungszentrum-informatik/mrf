#pragma once

#include <memory>
#include "camera_model.h"

namespace mrf {

/** @brief Determines support point and direction of the viewing ray of a given pixel
 *  Will always return true */
struct CameraModelOrthoGetViewingRay {
    template <typename T, typename T1, typename T2, typename T3>
    bool operator()(T, T1&& imagePoint, T2&& pos, T3&& direction) const {
        direction[0] = 0;
        direction[1] = 0;
        direction[2] = T(1);
        pos[0] = imagePoint[0];
        pos[1] = imagePoint[1];
        pos[2] = T(0);
        return true;
    }
};

/** @brief Determines if the 3D point is in front of an image point.
 *  Will always return true */
struct CameraModelOrthoGetImagePoint {
    template <typename T, typename T1, typename T2>
    inline bool operator()(T, T1&& point3d, T2&& imagePoint) const {
        imagePoint[0] = point3d[0];
        imagePoint[1] = point3d[1];
        return true;
    }
};

/** @brief Orthographic camera class */
class CameraModelOrtho : public CameraModel {

public:
    inline CameraModelOrtho(){};
    inline CameraModelOrtho(int imgWidth, int imgHeight)
            : imgWidth_{imgWidth}, imgHeight_{imgHeight} {};
    inline virtual ~CameraModelOrtho(){};
    /** @brief Get the image size
     * @param imgWidth Image width
     * @param imgHeight Image height */
    inline virtual void getImageSize(int& imgWidth, int& imgHeight) const override {
        imgWidth = imgWidth_;
        imgHeight = imgHeight_;
    }

    /** @brief Determines if a 3D point is in front of the image.
     *  @param point3d 3D point to check
     *  @param imagePoint Image point. Will have the according 3D point value if it's in front
     *  @return Always true */
    inline virtual bool getImagePoint(const Eigen::Ref<const Eigen::Vector3d>& point3d,
                                      Eigen::Ref<Eigen::Vector2d> imagePoint) const override {
        CameraModelOrthoGetImagePoint getImagePoint;
        return getImagePoint(double(), point3d, imagePoint);
    }

    /** @brief Get viewing ray from the camera origin to the pixel
     *  @param imagePoint Image point
     *  @param supportPoint Support point of the viewing ray
     *  @param direction Direction of the viewing ray
     *  @return Always true */
    inline virtual bool getViewingRay(const Eigen::Ref<const Eigen::Vector2d>& imagePoint,
                                      Eigen::Ref<Eigen::Vector3d> supportPoint,
                                      Eigen::Ref<Eigen::Vector3d> direction) const override {
        CameraModelOrthoGetViewingRay getViewingRay;
        return getViewingRay(double(), imagePoint, supportPoint, direction);
    }

    /** @brief Checks if this camera is SVP
     *  @return Always false */
    inline virtual bool isSvp() const override {
        return false;
    }

    /** @brief Get focal length of the camera
     *  @return Always 0 */
    inline virtual double getFocalLength() const override {
        return 0;
    }

    /** @brief Get principal point of camera
     *  @return Always zero vector */
    inline virtual Eigen::Vector2d getPrincipalPoint() const override {
        return Eigen::Vector2d::Zero();
    }

    /** @brief Clone this camera
     *  @return Pointer to cloned camera */
    inline virtual std::unique_ptr<CameraModel> clone() const override {
        return std::unique_ptr<CameraModelOrtho>(new CameraModelOrtho(*this));
    }

    /** @brief Get type of camera model
     *  @return CameraModelType::CAMERA_MODEL_PINHOLE */
    inline virtual CameraModelType getId() const override {
        return CameraModelType::CAMERA_MODEL_PINHOLE;
    }

private:
    int imgWidth_;  ///< Image width
    int imgHeight_; ///< Image height
};
}
