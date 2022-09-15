#include <cstdio>
#include <vector>

#include "DataFrame.h"
#include "DetectorTracker.h"
#include "DetectorLarge.h"

class SetupGeometry {
    
    public:

        SetupGeometry(std::string geometryFile);
        
	    void print();

        std::vector<DetectorTracker> detectorsTracker;
        std::vector<DetectorLarge> detectorsLarge;

    private:

        DataFrame geometryDataFrame;
};
