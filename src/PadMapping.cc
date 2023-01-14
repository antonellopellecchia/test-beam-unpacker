#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "PadMapping.h"
#include "DataFrame.h"

PadMapping::PadMapping(std::string mappingFilePath) {
	DataFrame mappingDataFrame = DataFrame::fromCsv(mappingFilePath);
	int chipId, chipChannel, eta, pad_x, pad_y;

	// iterate on rows:
	for (int irow=0; irow<mappingDataFrame.getNRows(); irow++) {
		// unused: oh = std::stoi(mappingRow[columnIndex["oh"]]);
		// unused: chamber = std::stoi(mappingRow[columnIndex["chamber"]]);
		chipId = std::stoi(mappingDataFrame.getElement("chip", irow));
		chipChannel = std::stoi(mappingDataFrame.getElement("channel", irow));
		pad_x = std::stoi(mappingDataFrame.getElement("pad_x", irow));
		pad_y = std::stoi(mappingDataFrame.getElement("pad_y", irow));
		to_pad_x[chipId][chipChannel] = pad_x;
		to_pad_y[chipId][chipChannel] = pad_y;
	}
}

void PadMapping::print() {
	std::cout << "Pad mapping" << std::endl;
  std::cout << "chip\tchannel\tpad x\tpad y" << std::endl;
	for (int i=0; i<3; i++) {
		for (int j=0; j<10; j++)
      std::cout << i << "\t" << j << "\t" << to_pad_x[i][j] << "\t" << to_pad_y[i][j] << std::endl;
		std::cout << "..." << std::endl;
	}
	std::cout << std::endl;
}
