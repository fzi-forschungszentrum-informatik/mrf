#pragma once

#include <memory>
#include <camera_models/camera_model.h>

namespace mrf {

struct CameraModelOrthoGetViewingRay {
    template <typename T, typename T1, typename T2, typename T3>
    bool operator()(T, T1&& imagePoint, T2&& pos, T3&& direction) const {
        direction[0] = 0;
        direction[1] = 0;
        direction[2] = T(1.0);
        pos[0] = pos[1] = pos[2] = T(0.0);
        return true;
    }
};
struct CameraModelOrthoGetImagePoint {
    template <typename T, typename T1, typename T2>
    bool operator()(T, T1&& point3d, T2&& imagePoint) const {
        imagePoint[0] = point3d[0];
        imagePoint[1] = point3d[1];
        return true;
    }
};

class CameraModelOrtho : public CameraModel {

public:
    inline CameraModelOrtho(){};
    inline CameraModelOrtho(int imgWidth, int imgHeight)
            : imgWidth_{imgWidth}, imgHeight_{imgHeight} {};
    inline virtual ~CameraModelOrtho(){};

    inline virtual void getImageSize(int& imgWidth, int& imgHeight) const override {
        imgWidth = imgWidth_;
        imgHeight = imgHeight_;
    }

    inline virtual bool getImagePoint(const Eigen::Ref<const Eigen::Vector3d>& point3d,
                                      Eigen::Ref<Eigen::Vector2d> imagePoint) const override {
        CameraModelOrthoGetImagePoint getImagePoint;
        return getImagePoint(double(), point3d, imagePoint);
    }

    inline virtual bool getViewingRay(const Eigen::Ref<const Eigen::Vector2d>& imagePoint,
                                      Eigen::Ref<Eigen::Vector3d> supportPoint,
                                      Eigen::Ref<Eigen::Vector3d> direction) const override {
        CameraModelOrthoGetViewingRay getViewingRay;
        return getViewingRay(double(), imagePoint, supportPoint, direction);
    }

    inline virtual bool isSvp() const override {
        return false;
    }
    inline virtual double getFocalLength() const override {
        return 0;
    }
    inline virtual Eigen::Vector2d getPrincipalPoint() const override {
        return Eigen::Vector2d::Zero();
    }

    inline virtual std::unique_ptr<CameraModel> clone() const override {
        return std::unique_ptr<CameraModelOrtho>(new CameraModelOrtho(*this));
    }

    inline virtual void doLoad(cereal::PortableBinaryInputArchive&) override{};
    inline virtual void doSave(cereal::PortableBinaryOutputArchive&) const override{};

    inline virtual CameraModelType getId() const override {
        return CameraModelType::CAMERA_MODEL_PINHOLE;
    }

private:
    int imgWidth_;
    int imgHeight_;
};
}
