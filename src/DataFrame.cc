#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "DataFrame.h"

DataFrame::DataFrame(std::vector<std::string> colNames, std::map<std::string, std::vector<std::string>> elements) {
	fColumnNames = colNames;
	fElements = elements;
}

DataFrame DataFrame::fromCsv(std::string path) {
	// parse csv mapping file
	std::string csvLine, buffer;
	std::ifstream csvFile(path);
	std::vector<std::string> mappingRow;

	std::vector<std::string> colNames;
	std::map<std::string, std::vector<std::string>> elements;
    
	if (!csvFile.is_open())
		throw std::invalid_argument("Could not open file "+path);
	for (int lineIndex=0; getline(csvFile, csvLine); lineIndex++) {
		mappingRow.clear();

		std::stringstream lineStream(csvLine);
		// split line by separator
		while (getline(lineStream, buffer, ',')) {
			mappingRow.push_back(buffer);
		}

		if (lineIndex==0) { // parse header
			colNames = mappingRow;
		} else { // parse mapping row
			for (int icol=0; icol<mappingRow.size(); icol++) {
				// in case of generic class:
				// std::stringstream convertToT(mappingRow[icol]);
				// T value;
				// convertToT >> value;
				// fElements[colNames[icol]].push_back(value);
				elements[colNames[icol]].push_back(mappingRow[icol]);
			}
		}
	}
	csvFile.close();
	return DataFrame(colNames, elements);
}

void DataFrame::print() {
	for (auto colName:fColumnNames) std::cout << colName << "\t";
	std::cout << std::endl;

	int minRows = getNRows();
	if (minRows>10) minRows = 10;
	for (int irow=0; irow<minRows; irow++) {
		for (auto colName:fColumnNames) std::cout << fElements[colName][irow] << "\t";
		std::cout << std::endl;
	}
}
