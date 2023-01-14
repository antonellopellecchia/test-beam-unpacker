#include <cstdio>
#include <string>

#ifndef DEF_STRIPMAPPING
#define DEF_STRIPMAPPING

class PadMapping {
    
    public:

	    PadMapping(std::string mappingFilePath);

	    int read();
	    void print();

	    int to_pad_x[24][128];
	    int to_pad_y[24][128];
};

#endif
