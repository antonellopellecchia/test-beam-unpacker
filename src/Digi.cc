#include "Digi.h"

Digi::Digi(int chamber, int eta, int strip, int vfat) {
    fChamber = chamber;
    fEta = eta;
    fStrip = strip;
    fVFAT = vfat;
}

int Digi::getChamber() { return fChamber; }
int Digi::getEta() { return fEta; }
int Digi::getStrip() { return fStrip; }
int Digi::getVFAT() {return fVFAT; }
