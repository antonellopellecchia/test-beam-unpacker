#include <iostream>
#include <vector>

#include "Cluster.h"
#include "Digi.h"

Cluster::Cluster(int chamber, int eta, int first, int last, int vfat) {
    fChamber = chamber;
    fEta = eta;
    fFirst = first;
    fLast = last;
    fVFAT = vfat;
}

bool Cluster::isNeighbour(int strip) {
    return (strip==fFirst-1) || (strip==fLast+1);
}

void Cluster::extend(int strip) {
    if (strip < fFirst) fFirst = strip;
    else if (strip > fLast) fLast = strip;
}

double Cluster::getCenter() {
    return 0.5*(fFirst + fLast);
}

double Cluster::getSize() {
    return fLast - fFirst + 1;
}

int Cluster::getChamber() {return fChamber;}
int Cluster::getEta() {return fEta;}
int Cluster::getFirst() {return fFirst;}
int Cluster::getLast() {return fLast;}

int Cluster::getDirection() {
    return (fEta + 1) % 2;
}

int Cluster::getVFAT() {return fVFAT;}

std::vector<Cluster> Cluster::fromDigis(std::vector<Digi> digis) {
    std::vector<Cluster> clusters;

    int chamber;
    int eta;
    int VFAT;

    while (digis.size()>0) {
        Digi digi = digis[0];

        // Map OH, eta to chamber
        chamber = digi.getChamber();
        eta = digi.getEta();
        VFAT = digi.getVFAT();

        // Use as seed for cluster:
        Cluster cluster = Cluster(chamber, eta, digi.getStrip(), digi.getStrip(), VFAT);
        digis.erase(digis.begin());

        bool clusterUpdated = true;
        while (clusterUpdated) {
            clusterUpdated = false;
            for (int j=0; j<digis.size();) {
                if ((digis[j].getChamber() != cluster.getChamber()) || (digis[j].getEta() != cluster.getEta())) break;
                if (cluster.isNeighbour(digis[j].getStrip())) {
                    cluster.extend(digis[j].getStrip());
                    digis.erase(digis.begin() + j); // Don't increase j, list fill flow
                    // The cluster ends grew, so there might be more strips included
                    // Then scan the digi list again later:
                    clusterUpdated = true;
                } else j++;
            }
        }
        clusters.push_back(cluster);
    }
    return clusters;
}
