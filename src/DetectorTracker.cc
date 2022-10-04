#include <math.h>

#include "Cluster.h"
#include "Rechit.h"
#include "Rechit2D.h"
#include "DetectorTracker.h"

DetectorTracker::DetectorTracker(int oh, int chamber, double sizeX, double sizeY, int nStrips) {
    fOh = oh;
    fChamber = chamber;
    fSize[0] = sizeX;
    fSize[1] = sizeY;
    fNumberStrips = nStrips;
    fStripPitch = sizeX / fNumberStrips;
    fStripPitchSqrt12 = fStripPitch * 0.288675;
}

Rechit DetectorTracker::createRechit(Cluster cluster) {
    return Rechit(
        fChamber,
        fScale[cluster.getDirection()]*(-0.5*fSize[cluster.getDirection()] + fStripPitch*cluster.getCenter()),
        cluster.getSize() * fStripPitchSqrt12,
        cluster.getSize()
    );
}

Rechit2D DetectorTracker::createRechit2D(Cluster cluster1, Cluster cluster2) {
    return Rechit2D(fChamber, createRechit(cluster1), createRechit(cluster2));
}

void DetectorTracker::mapRechit2D(Rechit2D *rechit) {
    // map already existing rechit to global detector geometry
    double localX = rechit->getLocalX();
    double localY = rechit->getLocalY();
    rechit->setGlobalPosition(
        fPosition[0] + localX*cos(fTheta) - localY*sin(fTheta),
        fPosition[1] + localX*sin(fTheta) + localY*cos(fTheta),
        fPosition[2]
    );
}
