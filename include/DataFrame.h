#include <cstdio>

/*
    Parse csv file to map of vectors of type T
*/

#ifndef DATAFRAME_H
#define DATAFRAME_H

class DataFrame {
    
    public:

	    std::string fPath;

        DataFrame();
	    DataFrame(std::vector<std::string> colNames, std::map<std::string, std::vector<std::string>> elements);
	    
	    void print();

		int getNRows() { return fElements[fColumnNames[0]].size(); }
        bool contains(std::vector<std::string> keys);
		std::string getElement(std::string column, int row);

        bool isEmpty = false;
	    std::vector<std::string> fColumnNames;
	    std::map<std::string, std::vector<std::string>> fElements;
	    
		static DataFrame fromCsv(std::string path, std::string delimiter=",");
};

#endif
