#include <cstdio>
#include <iostream>
#include <cstdint>
#include <vector>
#include <array>
#include <bitset>
#include <signal.h>

#include <TFile.h>
#include <TTree.h>

#include "Digi.h"
#include "Cluster.h"
#include "DetectorTracker.h"
#include "DetectorLarge.h"
#include "Rechit2D.h"

#include "progressbar.h"

bool isInterrupted = false;
void interruptHandler(int dummy) {
    isInterrupted = true;
}

int main (int argc, char** argv) {

  if (argc<3) {
    std::cout << "Usage: DigiToRechits ifile ofile [--verbose] [--events n] [--geometry may2022/july2022]" << std::endl;
    return 0;
  }
  std::string ifile   = argv[1];
  std::string ofile   = argv[2];
    
  int max_events = -1;
  bool verbose = false;
  std::string geometry;
  for (int iarg=0; iarg<argc; iarg++) {
    std::string arg = argv[iarg];
    if (arg=="--verbose") verbose = true;
    else if (arg=="--events") max_events = atoi(argv[iarg+1]); 
    else if (arg=="--geometry") geometry = argv[iarg+1]; 
  }
 
  if (geometry == "") {
      std::cout << "Please specify a setup geometry" << std::endl;
      return -1;
  }

  if (max_events>=0) std::cout << "Analyzing " << max_events << " events" << std::endl;
  else std::cout << "Analyzing all events" << std::endl; 

  TFile digiFile(ifile.c_str(), "READ");
  TFile rechitFile(ofile.c_str(), "RECREATE", "Rechit tree file");
    
  TTree *digiTree = (TTree *) digiFile.Get("outputtree");
  TTree rechitTree("rechitTree","rechitTree");

  std::vector<DetectorTracker> detectorsTracker;
  std::vector<DetectorLarge> detectorsLarge;
  // define detector geometries
  if (geometry == "may2022") {
      detectorsTracker.push_back(DetectorTracker(2, 0, 89.5, 89.5, 358));
      detectorsTracker.push_back(DetectorTracker(2, 1, 89.5, 89.5, 358));
      detectorsTracker.push_back(DetectorTracker(3, 2, 89.5, 89.5, 358));
      detectorsTracker.push_back(DetectorTracker(3, 3, 89.5, 89.5, 358));
      detectorsLarge.push_back(DetectorLarge(0, 4, 488.8, 628.8, 390.9, 4, 384)); // ge21
      detectorsLarge.push_back(DetectorLarge(0, 5, 235.2, 460, 787.9, 8, 384)); // me0 blank
      detectorsLarge.push_back(DetectorLarge(1, 6, 235.2, 460, 787.9, 8, 384)); // me0 random
  } else if (geometry == "july2022") {
      detectorsTracker.push_back(DetectorTracker(0, 0, 100., 100., 256));
      detectorsTracker.push_back(DetectorTracker(0, 1, 100., 100., 256));
      detectorsTracker.push_back(DetectorTracker(0, 2, 100., 100., 256));
      detectorsLarge.push_back(DetectorLarge(0, 3, 235.2, 460, 787.9, 8, 384)); // me0 blank
  } else {
      std::cout << "Geometry \"" << geometry << "\" not supported." << std::endl;
      return -1;
  }

  int nTrackers = detectorsTracker.size();

  // digi variables
  int nhits;
  int orbitNumber, bunchCounter, eventCounter;
  std::vector<int> *vecDigiChamber = new std::vector<int>(); // 0 to 3 for trackers, 4 and 5 for GE21 and ME0
  std::vector<int> *vecDigiEta = new std::vector<int>(); // even for x, odd for y
  std::vector<int> *vecDigiDirection = new std::vector<int>(); // 0 for x, 1 for y
  std::vector<int> *vecDigiStrip = new std::vector<int>(); // 0 to 357
  std::vector<int> *vecRawChannel = new std::vector<int>();

  // cluster variables
  int nclusters;
  std::vector<int> vecClusterChamber;
  std::vector<int> vecClusterEta;
  std::vector<int> vecClusterCenter;
  std::vector<int> vecClusterFirst;
  std::vector<int> vecClusterSize;

  // rechits variables
  int nrechits;
  std::vector<int> vecRechitChamber;
  std::vector<int> vecRechitEta;
  std::vector<double> vecRechitX;
  std::vector<double> vecRechitY;
  std::vector<double> vecRechitError;
  std::vector<double> vecRechitClusterSize;

  // rechits 2d variables
  int nrechits2d;
  std::vector<int> vecRechit2DChamber;
  std::vector<double> vecRechit2D_X_Center;
  std::vector<double> vecRechit2D_Y_Center;
  std::vector<double> vecRechit2D_X_Error;
  std::vector<double> vecRechit2D_Y_Error;
  std::vector<double> vecRechit2D_X_ClusterSize;
  std::vector<double> vecRechit2D_Y_ClusterSize;

  // support variables
  int oh, eta;
  int chamber, chamber1, chamber2;
  int direction1, direction2;
  Rechit rechit;
  Rechit2D rechit2D;

  std::vector<Digi> digisInEvent;
  std::vector<Cluster> clustersInEvent;

  // digi variable branches
  digiTree->SetBranchAddress("orbitNumber", &orbitNumber);
  digiTree->SetBranchAddress("bunchCounter", &bunchCounter);
  digiTree->SetBranchAddress("eventCounter", &eventCounter);
  digiTree->SetBranchAddress("nhits", &nhits);
  digiTree->SetBranchAddress("digiChamber", &vecDigiChamber);
  digiTree->SetBranchAddress("digiEta", &vecDigiEta);
  digiTree->SetBranchAddress("digiStrip", &vecDigiStrip);
  digiTree->SetBranchAddress("CH", &vecRawChannel);
  //digiTree->SetBranchAddress("digiDirection", &vecDigiDirection);

  // event branches
  rechitTree.Branch("orbitNumber", &orbitNumber);
  rechitTree.Branch("eventCounter", &eventCounter);
  rechitTree.Branch("bunchCounter", &bunchCounter);

  // cluster branches
  rechitTree.Branch("nclusters", &nclusters, "nclusters/I");
  rechitTree.Branch("clusterChamber", &vecClusterChamber);
  rechitTree.Branch("clusterEta", &vecClusterEta);
  rechitTree.Branch("clusterCenter", &vecClusterCenter);
  rechitTree.Branch("clusterFirst", &vecClusterFirst);
  rechitTree.Branch("clusterSize", &vecClusterSize);

  // rechit branches
  rechitTree.Branch("nrechits", &nrechits, "nrechits/I");
  rechitTree.Branch("rawChannel", vecRawChannel);
  rechitTree.Branch("digiStrip", vecDigiStrip);
  rechitTree.Branch("rechitChamber", &vecRechitChamber);
  rechitTree.Branch("rechitEta", &vecRechitEta);
  rechitTree.Branch("rechitX", &vecRechitX);
  rechitTree.Branch("rechitY", &vecRechitY);
  rechitTree.Branch("rechitError", &vecRechitError);
  rechitTree.Branch("rechitClusterSize", &vecRechitClusterSize);

  // rechit2D branches
  rechitTree.Branch("nrechits2d", &nrechits2d, "nrechits2d/I");
  rechitTree.Branch("rechit2DChamber", &vecRechit2DChamber);
  rechitTree.Branch("rechit2D_X_center", &vecRechit2D_X_Center);
  rechitTree.Branch("rechit2D_Y_center", &vecRechit2D_Y_Center);
  rechitTree.Branch("rechit2D_X_error", &vecRechit2D_X_Error);
  rechitTree.Branch("rechit2D_Y_error", &vecRechit2D_Y_Error);
  rechitTree.Branch("rechit2D_X_clusterSize", &vecRechit2D_X_ClusterSize);
  rechitTree.Branch("rechit2D_Y_clusterSize", &vecRechit2D_Y_ClusterSize);

  int nentries = digiTree->GetEntries();
  std::cout << nentries << " total events" <<  std::endl;
  if (max_events>=0) {
    std::cout << "Processing " << nentries << " events" << std::endl;
    nentries = max_events;
  }
  progressbar bar(nentries);

  signal(SIGINT, interruptHandler);
  for (int nevt=0; (!isInterrupted) && digiTree->LoadTree(nevt)>=0; ++nevt) {
    if ((max_events>=0) && (nevt>=max_events)) break;
    
    if (verbose) std::cout << "Event " << nevt << "/" << nentries << std::endl;
    else bar.update();
    //if ( nevt%1000==0 ) std::cout << "Unpacking event " << nevt << "\t\t\t\r";

    digiTree->GetEntry(nevt);

    vecClusterChamber.clear();
    vecClusterEta.clear();
    vecClusterCenter.clear();
    vecClusterFirst.clear();
    vecClusterSize.clear();

    nrechits = 0;
    vecRechitChamber.clear();
    vecRechitEta.clear();
    vecRechitX.clear();
    vecRechitY.clear();
    vecRechitError.clear();
    vecRechitClusterSize.clear();

    nrechits2d = 0;
    vecRechit2DChamber.clear();
    vecRechit2D_X_Center.clear();
    vecRechit2D_Y_Center.clear();
    vecRechit2D_X_Error.clear();
    vecRechit2D_Y_Error.clear();
    vecRechit2D_X_ClusterSize.clear();
    vecRechit2D_Y_ClusterSize.clear();

    digisInEvent.clear();
    for (int ihit=0; ihit<nhits; ihit++)
      digisInEvent.push_back(Digi(
        vecDigiChamber->at(ihit),
        vecDigiEta->at(ihit),
        vecDigiStrip->at(ihit)
      ));
    clustersInEvent = Cluster::fromDigis(digisInEvent);

    nclusters = clustersInEvent.size();
    for (int icluster=0; icluster<nclusters; icluster++) {
      vecClusterChamber.push_back(clustersInEvent[icluster].getChamber());
      vecClusterEta.push_back(clustersInEvent[icluster].getEta());
      vecClusterCenter.push_back(clustersInEvent[icluster].getCenter());
      vecClusterFirst.push_back(clustersInEvent[icluster].getFirst());
      vecClusterSize.push_back(clustersInEvent[icluster].getSize());

      if (clustersInEvent[icluster].getChamber() >= nTrackers) {
        // for large chamber, build 1D rechits:
        int chamber = clustersInEvent[icluster].getChamber();
        //rechit = Rechit(chamber, 0, clustersInEvent[icluster]);
        if (verbose) {
            std::cout << "  Chamber " << chamber;
            std::cout << " eta " << clustersInEvent[icluster].getEta();
        }
        // create rechit from cluster on chosen detector:
        for (DetectorLarge detector:detectorsLarge) {
            if (detector.getChamber() == chamber) rechit = detector.createRechit(clustersInEvent[icluster]);
        }
        vecRechitChamber.push_back(chamber);
        vecRechitEta.push_back(clustersInEvent[icluster].getEta());
        vecRechitX.push_back(rechit.getCenter());
        vecRechitY.push_back(rechit.getY());
        vecRechitError.push_back(rechit.getError());
        vecRechitClusterSize.push_back(rechit.getClusterSize());
        nrechits++;
        if (verbose) {
            std::cout << " local (" << rechit.getCenter() << ",";
            std::cout << rechit.getY() << ")" << std::endl;
        }
      } else {
        // for tracker, build 2D rechits:
        chamber1 = clustersInEvent[icluster].getChamber();
        direction1 = clustersInEvent[icluster].getDirection();
        if (direction1 != 0) continue; // first cluster in X direction

        for (int jcluster=0; jcluster<nclusters; jcluster++) {
          // match with all clusters in perpendicular direction
          if (clustersInEvent[icluster].getChamber() != clustersInEvent[jcluster].getChamber()) continue;
          
          chamber2 = clustersInEvent[jcluster].getChamber();
          if (chamber1!=chamber2) continue;
          direction2 = clustersInEvent[jcluster].getDirection();
          if (direction1==direction2) continue;

          //rechit2D = Rechit2D(chamber1, clustersInEvent[icluster], clustersInEvent[jcluster]);
          rechit2D = detectorsTracker[chamber1].createRechit2D(clustersInEvent[icluster], clustersInEvent[jcluster]);

          vecRechit2DChamber.push_back(chamber1);
          vecRechit2D_X_Center.push_back(rechit2D.getLocalX());
          vecRechit2D_Y_Center.push_back(rechit2D.getLocalY());
          vecRechit2D_X_Error.push_back(rechit2D.getErrorX());
          vecRechit2D_Y_Error.push_back(rechit2D.getErrorY());
          vecRechit2D_X_ClusterSize.push_back(rechit2D.getClusterSizeX());
          vecRechit2D_Y_ClusterSize.push_back(rechit2D.getClusterSizeY());
          nrechits2d++;

          if (verbose) {
            std::cout << "  Chamber " << chamber1;
            std::cout << " local (" << rechit2D.getLocalX() << ",";
            std::cout << rechit2D.getLocalY() << ")" << std::endl;
          }
        }
      }
    }
    
    rechitTree.Fill();
  }
  std::cout << std::endl;

  rechitTree.Write();
  rechitFile.Close();
  std::cout << "Output file saved to " << ofile << std::endl;
}
