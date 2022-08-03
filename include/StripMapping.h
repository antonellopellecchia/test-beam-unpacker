#include <cstdio>
#include <string>

#ifndef DEF_STRIPMAPPING
#define DEF_STRIPMAPPING

class StripMapping {
    
    public:

	    StripMapping(std::string mappingFilePath);

	    int read();
	    void print();

	    int to_strip[24][128];
	    int to_eta[24] = {0};
};

#endif
