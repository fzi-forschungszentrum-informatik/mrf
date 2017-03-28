#include "camera.h"

// Camera Camera::clone() {
//	Camera camOut;
//	camOut.referenceFrame = this->referenceFrame;
//	camOut.cameraPose = this->cameraPose;
//	camOut.cameraModel = this->cameraModel->clone();
//	return camOut;
//}

std::ostream& operator<<(std::ostream& out, Camera& cam) {
    out << *cam.cameraModel.get();
    out << "reference frame: " << cam.referenceFrame << std::endl;
    out << "pose:" << std::endl;
    out << cam.cameraPose.matrix() << std::endl;
    return out;
}
