#pragma once

#include <string>
#include <Eigen/Eigen>
#include "camera_model.h"

class Camera {
public:
    std::string referenceFrame;
    Eigen::Affine3d cameraPose; // transformation from camera to referenceFrame
    std::unique_ptr<CameraModel> cameraModel;

    Camera(){};
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera& c) = delete;

    Camera(Camera&& c)
            : referenceFrame(std::move(c.referenceFrame)), cameraPose(std::move(c.cameraPose)),
              cameraModel(std::move(c.cameraModel)) {
    }
    Camera& operator=(Camera&& c) {
        if (&c != this) {
            referenceFrame = std::move(c.referenceFrame);
            cameraPose = std::move(c.cameraPose);
            cameraModel = std::move(c.cameraModel);
        }
        return *this;
    }

    Camera clone();
};

std::ostream& operator<<(std::ostream& out, Camera& cam);
