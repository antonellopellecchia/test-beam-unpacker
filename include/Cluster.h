#include <vector>

#include "Digi.h"

#ifndef DEF_CLUSTER
#define DEF_CLUSTER

class Cluster {

    public:

        int fChamber, fEta, fVFAT;
        int fFirst, fLast;

        Cluster() {}
        Cluster(int chamber, int eta, int first, int last, int vfat=99);

        bool isNeighbour(int strip);
        void extend(int strip);
        double getCenter();
        double getSize();

        int getChamber();
        int getEta();
        int getFirst();
        int getLast();
        int getDirection();
        int getVFAT();

        static std::vector<Cluster> fromDigis(std::vector<Digi> digis);
    
        std::ostream& operator<<(std::ostream &os) {
            return os << fChamber << "chamber." << fEta << "eta." << fFirst << "first." << fLast << "last";
        }
        
        void print() {
            std::cout << fChamber << "chamber." << fEta << "eta." << fFirst << "first." << fLast << "last" << std::endl;
        }
};

#endif
