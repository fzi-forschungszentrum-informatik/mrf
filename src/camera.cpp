#include "camera.h"

std::ostream& operator<<(std::ostream& out, Camera& cam) {
    out << *cam.cameraModel.get();
    out << "reference frame: " << cam.referenceFrame << std::endl;
    out << "pose:" << std::endl;
    out << cam.cameraPose.matrix() << std::endl;
    return out;
}
