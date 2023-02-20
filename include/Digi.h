#include <iostream>

#ifndef DEF_DIGI
#define DEF_DIGI

class Digi{
    
    public:
        int fChamber, fEta, fStrip, fVFAT;

        Digi() {}
        Digi(int chamber, int eta, int strip, int vfat=99);

        int getChamber();
        int getEta();
        int getStrip();
        int getVFAT();

        void print() {
            std::cout << fChamber << "chamber." << fEta << "eta." << fStrip << "strip" << std::endl;
        }
};

#endif
