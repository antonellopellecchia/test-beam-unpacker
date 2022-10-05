#include <math.h>

#include "Hit.h"
#include "DetectorLarge.h"

Hit::Hit(double x, double y, double z, double errX, double errY, double errZ) {
    fGlobalPosition[0] = x;
    fGlobalPosition[1] = y;
    fGlobalPosition[2] = z;
    fErrPosition[0] = errX;
    fErrPosition[1] = errY;
    fErrPosition[2] = errZ;
}

Hit::Hit(DetectorGeometry *detector, double x, double y, double z, double errX, double errY, double errZ) :
    Hit(x, y, z, errX, errY, errZ) {
    setDetector(detector);
}

Hit Hit::fromLocal(DetectorGeometry *detector, double localX, double localY, double errX, double errY, double errZ) {
    // std::cout << "              ######### DEBUGGING INFO #########" << std::endl;
    // std::cout << "              local\t\tdet\t\tglobal" << std::endl;
    // std::cout << "              " << localX << "\t\t" << detector->getPositionX() << "\t\t" << localX+detector->getPositionX() << std::endl;
    // std::cout << "              " << localY << "\t\t" << detector->getPositionY() << "\t\t" << localY+detector->getPositionY() << std::endl;
    // std::cout << "              " << detector->getPositionZ() << std::endl;
    // std::cout << "              ######### END DEBUGGING #########" << std::endl;
    
    // calculate global coordinates:
    double globalX = detector->getPositionX() + localX*cos(detector->getTheta()) - localY*sin(detector->getTheta());
    double globalY = detector->getPositionY() + localX*sin(detector->getTheta()) + localY*cos(detector->getTheta());

    return Hit(
        detector,
        globalX, globalY, detector->getPositionZ(),
        errX, errY, errZ
    );
}

void Hit::setDetector(DetectorGeometry *detector) {
    fDetector = detector;
    // calculate local coordinates:
    double x = fGlobalPosition[0] - detector->getPositionX();
    double y = fGlobalPosition[1] - detector->getPositionY();
    fLocalPosition[0] = x*cos(detector->getTheta()) + y*sin(detector->getTheta());
    fLocalPosition[1] = -x*sin(detector->getTheta()) + y*cos(detector->getTheta());
    fLocalR = sqrt( pow(fLocalPosition[0],2) + pow(fLocalPosition[1]-detector->getOriginY(),2) );
    fLocalPhi = atan( fLocalPosition[0]/(fLocalPosition[1]-detector->getOriginY()) );
}

int Hit::getEta() {
    // calculate eta partition of hit from detector geometry    
    return fDetector->getNEta() - floor(getLocalY()/fDetector->getEtaHeight());
}
