#include <vector>

#include "DetectorGeometry.h"
#include "Rechit.h"

#ifndef DEF_DETECTORLARGE
#define DEF_DETECTORLARGE

class DetectorLarge : public DetectorGeometry {

    public:

        DetectorLarge(int oh, int chamber, double baseNarrow, double baseWide, double height, int nEta, int nStrips);

        double getY(int eta);
        double getYTop(int eta);
        double getWidth(int eta);
        double getStripPitch(int eta);
        double getStripPitchSqrt12(int eta);
        
        double getChamber() { return fChamber; }
        double getBaseNarrow() { return fBaseNarrow; }
        double getBaseWide() { return fBaseWide; }
        double getHeight() { return fHeight; }
        double getNStrips() { return fNumberStrips; }

        Rechit createRechit(Cluster cluster);
        void mapRechit(Rechit *rechit);
    
    private:

        int fOh, fChamber;
        double fBaseNarrow, fBaseWide, fHeight;
        int fNumberStrips;

        std::vector<double> fPartitionYs;
        std::vector<double> fPartitionYTops;
        std::vector<double> fPartitionWidths;
        std::vector<double> fPartitionStripPitches;
        std::vector<double> fPartitionStripPitchesSqrt12;
};

#endif
