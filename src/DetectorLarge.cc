#include <math.h>

#include "Cluster.h"
#include "Rechit.h"
#include "DetectorLarge.h"

DetectorLarge::DetectorLarge(int oh, int chamber, double baseNarrow, double baseWide, double height, int nEta, int nStrips) {
    fOh = oh;
    fChamber = chamber;
    fBaseNarrow = baseNarrow;
    fBaseWide = baseWide;
    fHeight = height;
    fNumberPartitions = nEta;
    fNumberStrips = nStrips;
    fEtaHeight = height/nEta;

    std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Initializing large chamber " << chamber << " with oh " << oh << std::endl;
    std::cout << "Narrow base " << baseNarrow << ", wide base " << baseWide << ", height " << height << std::endl;
    std::cout << "Eta partitions " << nEta << ", strips " << nStrips << ", eta partition height " << fEtaHeight << std::endl;

    fPartitionYs.reserve(fNumberPartitions);
    fPartitionYTops.reserve(fNumberPartitions);
    fPartitionWidths.reserve(fNumberPartitions);
    fPartitionStripPitches.reserve(fNumberPartitions);
    fPartitionStripPitchesSqrt12.reserve(fNumberPartitions);
    for (int eta=0; eta<fNumberPartitions; eta++) {
        fPartitionYs.push_back(fEtaHeight*(0.5 + (double)(fNumberPartitions-eta-1)));
        fPartitionYTops.push_back(fPartitionYs[eta] + 0.5*fEtaHeight);
        fPartitionWidths.push_back(fBaseNarrow + fPartitionYs[eta]*(fBaseWide-fBaseNarrow)/fHeight);
        fPartitionStripPitches.push_back(fPartitionWidths[eta] / fNumberStrips);
        fPartitionStripPitchesSqrt12.push_back(fPartitionStripPitches[eta] * 0.288675);

        std::cout << "    eta partition " << eta+1;
        std::cout << ", middle y " << fPartitionYs[eta] << ", width " << fPartitionWidths[eta];
        std::cout << ", strip pitch " << fPartitionStripPitches[eta];
        std::cout << ", expected resolution " << fPartitionStripPitchesSqrt12[eta] << std::endl;
    }

    // calculate and print detector geometric parameters:
    fOriginY = baseWide*height/(baseWide-baseNarrow);
    fArea = 0.5*(baseWide+baseNarrow)*height;
    fAperture = 2*atan(0.5*baseWide/fOriginY);
    std::cout << "Radius " << fOriginY;
    std::cout << ", area " << fArea;
    std::cout << ", aperture " << fAperture;
    std::cout << ", pitch " << fAperture/fNumberStrips;
    std::cout << ", expected resolution " << fAperture/fNumberStrips/pow(12, 0.5) << std::endl;

    std::cout << std::endl;
}

double DetectorLarge::getY(int eta) {
    return fPartitionYs[eta-1];
}

double DetectorLarge::getYTop(int eta) {
    return fPartitionYTops[eta-1];
}

double DetectorLarge::getWidth(int eta) {
    return fPartitionWidths[eta-1];
}

double DetectorLarge::getStripPitch(int eta) {
    return fPartitionStripPitches[eta-1];
}

double DetectorLarge::getStripPitchSqrt12(int eta) {
    return fPartitionStripPitchesSqrt12[eta-1];
}

Rechit DetectorLarge::createRechit(Cluster cluster) {
    Rechit rechit(
        fChamber,
        -0.5*getWidth(cluster.getEta()) + getStripPitch(cluster.getEta()) * cluster.getCenter(),
        cluster.getSize() * getStripPitchSqrt12(cluster.getEta()),
        cluster.getSize()
    );
    rechit.setY(getY(cluster.getEta()));
    return rechit;
}

void DetectorLarge::mapRechit(Rechit *rechit) {
    // map already existing rechit to global detector geometry
    double localX = rechit->getCenter();
    rechit->setGlobalPosition(
        fPosition[0] + localX*cos(fTheta),// - localY*sin(fTheta),
        fPosition[2]
    );
}
