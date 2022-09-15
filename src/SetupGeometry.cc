#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <stdexcept>

#include "SetupGeometry.h"
#include "DataFrame.h"

SetupGeometry::SetupGeometry(std::string geometryFile) {

	geometryDataFrame = DataFrame::fromCsv(geometryFile);
    std::cout << "Read dataframe" << std::endl;
    geometryDataFrame.print();

    std::string type;

	// iterate on detectors:
	for (int idetector=0; idetector<geometryDataFrame.getNRows(); idetector++) {

        type = geometryDataFrame.getElement("type", idetector);
        if(type == "tracker") {
            detectorsTracker.push_back(DetectorTracker(
                0, // OH number is always 0, unused
                std::stoi(geometryDataFrame.getElement("chamber", idetector)),
                std::stof(geometryDataFrame.getElement("width", idetector)),
                std::stof(geometryDataFrame.getElement("height", idetector)),
                std::stoi(geometryDataFrame.getElement("strips", idetector))
            ));
            detectorsTracker[detectorsTracker.size()-1].setPosition(
                std::stof(geometryDataFrame.getElement("x", idetector)),
                std::stof(geometryDataFrame.getElement("y", idetector)),
                std::stof(geometryDataFrame.getElement("z", idetector)),
                std::stof(geometryDataFrame.getElement("angle", idetector))
            );
        } else if(type == "large") {
            detectorsLarge.push_back(DetectorLarge(
                0, // OH number is always 0, unused
                std::stoi(geometryDataFrame.getElement("chamber", idetector)),
                std::stof(geometryDataFrame.getElement("width", idetector)),
                std::stof(geometryDataFrame.getElement("widthLarge", idetector)),
                std::stof(geometryDataFrame.getElement("height", idetector)),
                std::stoi(geometryDataFrame.getElement("eta", idetector)),
                std::stoi(geometryDataFrame.getElement("strips", idetector))
            ));
            detectorsLarge[detectorsLarge.size()-1].setPosition(
                std::stof(geometryDataFrame.getElement("x", idetector)),
                std::stof(geometryDataFrame.getElement("y", idetector)),
                std::stof(geometryDataFrame.getElement("z", idetector)),
                std::stof(geometryDataFrame.getElement("angle", idetector))
            );
        } else {
            throw std::invalid_argument(std::string("Invalid detector type '" + type + "' in geometry"));
        }
}
}

void SetupGeometry::print() {
    geometryDataFrame.print();
}
