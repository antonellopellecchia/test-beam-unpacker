#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <set>
#include <array>
#include <bitset>
#include <signal.h>
#include <math.h>
#include <sys/stat.h>

#include <chrono>
#include <thread>

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH2D.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>

#include "Digi.h"
#include "Cluster.h"
#include "Rechit.h"
#include "Rechit2D.h"
#include "Hit.h"
#include "Track2D.h"
#include "DetectorTracker.h"
#include "DetectorLarge.h"
#include "SetupGeometry.h"

#include "progressbar.h"

bool isInterrupted = false;
void interruptHandler(int dummy) {
    isInterrupted = true;
}

int main (int argc, char** argv) {

    if (argc<3) {
      std::cout << "Usage: Tracks ifile ofile [--geometry may2022/july2022] [--verbose] [--events n] [-x corrections x] [-y corrections y] [--angles correction angles]" << std::endl;
      return 0;
    }
    std::string ifile   = argv[1];
    std::string ofile   = argv[2];
    
    int max_events = -1;
    bool verbose = false;
    std::string geometry;
    double trackerAngles[4] = { .01158492576947, .00388093867016,  -.00190211740939, -.00971001194466 };
    double trackerCorrectionsX[4] = { -.68888635583799, .01851981215192, -.01738387261102, -.01552764211394 };
    double trackerCorrectionsY[4] = { -3.51064702409602, -1.08306067022753, .57358265976411, -.10031691598836 };
    for (int iarg=0; iarg<argc; iarg++) {
      std::string arg = argv[iarg];
      if (arg=="--verbose") verbose = true;
      else if (arg=="--events") max_events = atoi(argv[iarg+1]);
      else if (arg=="--geometry") geometry = argv[iarg+1];
      else if (arg=="--angles") {
        std::cout << "angles: ";
        for (int iangle=0; iangle<4; iangle++) {
          trackerAngles[iangle] = atof(argv[iarg+iangle+1]);
          std::cout << trackerAngles[iangle] << " ";
        }
        std::cout << std::endl;
      }
      else if (arg=="--x") {
        std::cout << "translation x: ";
        for (int ix=0; ix<4; ix++) {
          trackerCorrectionsX[ix] = atof(argv[iarg+ix+1]);
          std::cout << trackerCorrectionsX[ix] << " ";
        }
        std::cout << std::endl;
      }
      else if (arg=="--y") {
        std::cout << "translation y: ";
        for (int iy=0; iy<4; iy++) {
          trackerCorrectionsY[iy] = atof(argv[iarg+iy+1]);
          std::cout << trackerCorrectionsY[iy] << " ";
        }
        std::cout << std::endl;
      }
   }

    if (max_events > 0) std::cout << "Analyzing " << max_events << " events" << std::endl;
    else std::cout << "Analyzing all events" << std::endl; 

    TFile rechitFile(ifile.c_str(), "READ");     
    TTree *rechitTree = (TTree *) rechitFile.Get("rechitTree");

    TFile trackFile(ofile.c_str(), "RECREATE", "Track file");
    TTree trackTree("trackTree", "trackTree");
 
    std::vector<DetectorTracker> detectorsTracker;
    std::vector<DetectorLarge> detectorsLarge;

    std::string geometryCsvPath("geometry/"+geometry+".csv");
    std::ifstream geometryFile(geometryCsvPath);
    if (!geometryFile.good()) {
        std::cout << "Geometry \"" << geometry << "\" not supported." << std::endl;
        return -1;
    } else {
        // define detector geometries
        SetupGeometry setupGeometry(geometryCsvPath);
        detectorsTracker = setupGeometry.detectorsTracker;
        detectorsLarge = setupGeometry.detectorsLarge;
    }

    const int nTrackers = detectorsTracker.size();
    // create map from chamber number to large detector:
    std::map<int, DetectorGeometry*> detectorsMap;
    for (DetectorLarge detector:detectorsLarge) detectorsMap[detector.getChamber()] = &detector;
    
    // rechit variables
    int nrechits;
    int orbitNumber, bunchCounter, eventCounter;
    std::vector<int> *vecRechitChamber = new std::vector<int>();
    std::vector<int> *vecRechitEta = new std::vector<int>();
    std::vector<double> *vecRechitX = new std::vector<double>();
    std::vector<double> *vecRechitY = new std::vector<double>();
    std::vector<double> *vecRechitError = new std::vector<double>();
    std::vector<double> *vecRechitClusterSize = new std::vector<double>();
    std::vector<int> *vecClusterCenter = new std::vector<int>();
    std::vector<int> *vecDigiStrip = new std::vector<int>();
    std::vector<int> *vecRawChannel = new std::vector<int>();
    // rechit2D variables
    int nrechits2d;
    std::vector<int> *vecRechit2DChamber = new std::vector<int>();
    std::vector<double> *vecRechit2D_X_Center = new std::vector<double>();
    std::vector<double> *vecRechit2D_Y_Center = new std::vector<double>();
    std::vector<double> *vecRechit2D_X_Error = new std::vector<double>();
    std::vector<double> *vecRechit2D_Y_Error = new std::vector<double>();
    std::vector<double> *vecRechit2D_X_ClusterSize = new std::vector<double>();
    std::vector<double> *vecRechit2D_Y_ClusterSize = new std::vector<double>();
    
    // event branches
    rechitTree->SetBranchAddress("orbitNumber", &orbitNumber);
    rechitTree->SetBranchAddress("bunchCounter", &bunchCounter);
    rechitTree->SetBranchAddress("eventCounter", &eventCounter);
    
    // rechit branches
    rechitTree->SetBranchAddress("nrechits", &nrechits);
    rechitTree->SetBranchAddress("rechitChamber", &vecRechitChamber);
    rechitTree->SetBranchAddress("rechitEta", &vecRechitEta);
    rechitTree->SetBranchAddress("rechitX", &vecRechitX);
    rechitTree->SetBranchAddress("rechitY", &vecRechitY);
    rechitTree->SetBranchAddress("rechitError", &vecRechitError);
    rechitTree->SetBranchAddress("clusterCenter", &vecClusterCenter);
    rechitTree->SetBranchAddress("rechitClusterSize", &vecRechitClusterSize);
    rechitTree->SetBranchAddress("digiStrip", &vecDigiStrip);
    rechitTree->SetBranchAddress("rawChannel", &vecRawChannel);
    // rechit2D branches
    rechitTree->SetBranchAddress("nrechits2d", &nrechits2d);
    rechitTree->SetBranchAddress("rechit2DChamber", &vecRechit2DChamber);
    rechitTree->SetBranchAddress("rechit2D_X_center", &vecRechit2D_X_Center);
    rechitTree->SetBranchAddress("rechit2D_Y_center", &vecRechit2D_Y_Center);
    rechitTree->SetBranchAddress("rechit2D_X_error", &vecRechit2D_X_Error);
    rechitTree->SetBranchAddress("rechit2D_Y_error", &vecRechit2D_Y_Error);
    rechitTree->SetBranchAddress("rechit2D_X_clusterSize", &vecRechit2D_X_ClusterSize);
    rechitTree->SetBranchAddress("rechit2D_Y_clusterSize", &vecRechit2D_Y_ClusterSize);

    // track variables
    std::vector<double> tracks_X_chi2;
    std::vector<double> tracks_Y_chi2;
    std::vector<double> tracks_X_slope;
    std::vector<double> tracks_Y_slope;
    std::vector<double> tracks_X_intercept;
    std::vector<double> tracks_Y_intercept;
    std::vector<double> tracks_X_covariance;
    std::vector<double> tracks_Y_covariance;
    // rechit 2D variables
    std::vector<int> rechits2D_Chamber;
    std::vector<double> rechits2D_X;
    std::vector<double> rechits2D_Y;
    std::vector<double> rechits2D_X_Error;
    std::vector<double> rechits2D_Y_Error;
    std::vector<double> rechits2D_X_ClusterSize;
    std::vector<double> rechits2D_Y_ClusterSize;
    std::vector<double> prophits2D_X;
    std::vector<double> prophits2D_Y;
    std::vector<double> prophits2D_X_Error;
    std::vector<double> prophits2D_Y_Error;
    // rechit and prophit variables
    std::vector<int> rechitsChamber, prophitsChamber;
    double trackChi2X, trackChi2Y;
    double trackCovarianceX, trackCovarianceY;
    double trackSlopeX, trackSlopeY;
    double trackInterceptX, trackInterceptY;
    std::vector<double> allChi2;
    std::vector<double> rechitsEta;
    std::vector<double> rechitsLocalX;
    std::vector<double> rechitsLocalY;
    std::vector<double> rechitsLocalR;
    std::vector<double> rechitsLocalPhi;
    std::vector<double> rechitsGlobalX;
    std::vector<double> rechitsGlobalY;
    std::vector<double> rechitsClusterSize;
    std::vector<double> prophitsEta;
    std::vector<double> prophitsGlobalX;
    std::vector<double> prophitsGlobalY;
    std::vector<double> prophitsErrorX;
    std::vector<double> prophitsErrorY;
    std::vector<double> prophitsLocalX;
    std::vector<double> prophitsLocalY;
    std::vector<double> prophitsLocalR;
    std::vector<double> prophitsLocalPhi;

    int chamber;
    double rechitX, rechitY;
    double rechitX_clusterSize, rechitY_clusterSize;
    double prophitX, prophitY;
    double propErrorX, propErrorY;

    // event branches
    trackTree.Branch("orbitNumber", &orbitNumber);
    trackTree.Branch("bunchCounter", &bunchCounter);
    trackTree.Branch("eventCounter", &eventCounter);

    // track branches
    trackTree.Branch("tracks_X_chi2", &tracks_X_chi2);
    trackTree.Branch("tracks_Y_chi2", &tracks_Y_chi2);
    trackTree.Branch("tracks_X_slope", &tracks_X_slope);
    trackTree.Branch("tracks_Y_slope", &tracks_Y_slope);
    trackTree.Branch("tracks_X_intercept", &tracks_X_intercept);
    trackTree.Branch("tracks_Y_intercept", &tracks_Y_intercept);
    trackTree.Branch("tracks_X_covariance", &tracks_X_covariance);
    trackTree.Branch("tracks_Y_covariance", &tracks_Y_covariance);

    // rechit 2D branches
    trackTree.Branch("rechits2D_Chamber", &rechits2D_Chamber);
    trackTree.Branch("rechits2D_X", &rechits2D_X);
    trackTree.Branch("rechits2D_Y", &rechits2D_Y);
    trackTree.Branch("rechits2D_X_Error", &rechits2D_X_Error);
    trackTree.Branch("rechits2D_Y_Error", &rechits2D_Y_Error);
    trackTree.Branch("rechits2D_X_ClusterSize", &rechits2D_X_ClusterSize);
    trackTree.Branch("rechits2D_Y_ClusterSize", &rechits2D_Y_ClusterSize);
    trackTree.Branch("prophits2D_X", &prophits2D_X);
    trackTree.Branch("prophits2D_Y", &prophits2D_Y);
    trackTree.Branch("prophits2D_X_Error", &prophits2D_X_Error);
    trackTree.Branch("prophits2D_Y_Error", &prophits2D_Y_Error);

    // rechit and prophit branches
    trackTree.Branch("trackChi2X", &trackChi2X, "trackChi2X/D");
    trackTree.Branch("trackChi2Y", &trackChi2Y, "trackChi2Y/D");
    trackTree.Branch("trackCovarianceX", &trackCovarianceX, "trackCovarianceX/D");
    trackTree.Branch("trackCovarianceY", &trackCovarianceY, "trackCovarianceY/D");
    trackTree.Branch("allChi2", &allChi2);
    trackTree.Branch("trackSlopeX", &trackSlopeX, "trackSlopeX/D");
    trackTree.Branch("trackSlopeY", &trackSlopeY, "trackSlopeY/D");
    trackTree.Branch("trackInterceptX", &trackInterceptX, "trackInterceptX/D");
    trackTree.Branch("trackInterceptY", &trackInterceptY, "trackInterceptY/D");
    trackTree.Branch("rechitChamber", &rechitsChamber);
    trackTree.Branch("prophitChamber", &prophitsChamber);
    trackTree.Branch("rechitEta", &rechitsEta);
    trackTree.Branch("rechitClusterCenter", vecClusterCenter);
    trackTree.Branch("rechitDigiStrip", vecDigiStrip);
    trackTree.Branch("rechitRawChannel", vecRawChannel);
    trackTree.Branch("rechitLocalX", &rechitsLocalX);
    trackTree.Branch("rechitLocalY", &rechitsLocalY);
    trackTree.Branch("rechitLocalR", &rechitsLocalR);
    trackTree.Branch("rechitLocalPhi", &rechitsLocalPhi);
    trackTree.Branch("rechitGlobalX", &rechitsGlobalX);
    trackTree.Branch("rechitGlobalY", &rechitsGlobalY);
    trackTree.Branch("rechitClusterSize", &rechitsClusterSize);
    trackTree.Branch("prophitEta", &prophitsEta);
    trackTree.Branch("prophitGlobalX", &prophitsGlobalX);
    trackTree.Branch("prophitGlobalY", &prophitsGlobalY);
    trackTree.Branch("prophitErrorX", &prophitsErrorX);
    trackTree.Branch("prophitErrorY", &prophitsErrorY);
    trackTree.Branch("prophitLocalX", &prophitsLocalX);
    trackTree.Branch("prophitLocalY", &prophitsLocalY);
    trackTree.Branch("prophitLocalR", &prophitsLocalR);
    trackTree.Branch("prophitLocalPhi", &prophitsLocalPhi);

    Track2D track;
    std::vector<Track2D> trackerTracks; // all possible tracks built with all tracker, choose only one at the end
    Rechit rechit;
    Rechit2D rechit2d;
    Hit hit;

    int nentries = rechitTree->GetEntries();
    int nentriesGolden = 0, nentriesNice = 0;
    // support array to exclude events with more than one hit per tracker:
    std::vector<double> hitsPerTrackingChamber(nTrackers);

    std::cout << nentries << " total events" <<  std::endl;
    if (max_events>0) nentries = max_events;
    progressbar bar(nentries);
    signal(SIGINT, interruptHandler);
    for (int nevt=0; (!isInterrupted) && rechitTree->LoadTree(nevt)>=0; ++nevt) {
      if ((max_events>0) && (nevt>max_events)) break;

      if (verbose) std::cout << "Event " << nevt << "/" << nentries << std::endl;
      else bar.update();

      /* reset support variables */
      for (int i=0; i<nTrackers; i++) {
        hitsPerTrackingChamber[i] = 0;
      }
      /* reset branch variables */
      // tracker branches:
      tracks_X_chi2.clear();
      tracks_Y_chi2.clear();
      tracks_X_slope.clear();
      tracks_Y_slope.clear();
      tracks_X_intercept.clear();
      tracks_Y_intercept.clear();
      tracks_X_covariance.clear();
      tracks_Y_covariance.clear();
      rechits2D_Chamber.clear();
      rechits2D_X.clear();
      rechits2D_Y.clear();
      rechits2D_X_Error.clear();
      rechits2D_Y_Error.clear();
      rechits2D_X_ClusterSize.clear();
      rechits2D_Y_ClusterSize.clear();
      prophits2D_X.clear();
      prophits2D_Y.clear();
      prophits2D_X_Error.clear();
      prophits2D_Y_Error.clear();

      // large chamber branches:
      rechitsChamber.clear();
      prophitsChamber.clear();
      rechitsEta.clear();
      rechitsLocalX.clear();
      rechitsLocalY.clear();
      rechitsGlobalX.clear();
      rechitsGlobalY.clear();
      rechitsLocalR.clear();
      rechitsLocalPhi.clear();
      rechitsClusterSize.clear();
      prophitsEta.clear();
      prophitsGlobalX.clear();
      prophitsGlobalY.clear();
      prophitsErrorX.clear();
      prophitsErrorY.clear();
      prophitsLocalX.clear();
      prophitsLocalY.clear();
      prophitsLocalR.clear();
      prophitsLocalPhi.clear();

      rechitTree->GetEntry(nevt);

      // process event only if at most one 2D rechit per tracking chamber:
      bool isNiceEvent = true;
      for (int irechit=0; irechit<nrechits2d; irechit++) {
        chamber = vecRechit2DChamber->at(irechit);
        if (hitsPerTrackingChamber[chamber]>0) isNiceEvent = false;
        hitsPerTrackingChamber[chamber]++;
      }
      if (verbose) {
          for (int i=0; i<hitsPerTrackingChamber.size(); i++) {
              std::cout << "  " << hitsPerTrackingChamber[i] << " rechits in chamber " << i << std::endl;
          }
      }

      // skip building tracks with n-1 trackers if is not a nice event:
      if (isNiceEvent) {

          if (verbose) {
              std::cout << "  #### Extrapolation on tracker ####" << std::endl;
          }
          nentriesNice++;

          for (int testedChamber=0; testedChamber<nTrackers; testedChamber++) {
            track.clear();
            // loop over rechits and make track:
            for (int irechit=0; irechit<nrechits2d; irechit++) {
              chamber = vecRechit2DChamber->at(irechit);
              rechit2d = Rechit2D(chamber,
                Rechit(chamber, vecRechit2D_X_Center->at(irechit), vecRechit2D_X_Error->at(irechit), vecRechit2D_X_ClusterSize->at(irechit)),
                Rechit(chamber, vecRechit2D_Y_Center->at(irechit), vecRechit2D_Y_Error->at(irechit), vecRechit2D_Y_ClusterSize->at(irechit))
              );
              // apply global geometry:
              detectorsTracker[chamber].mapRechit2D(&rechit2d);
              if (chamber!=testedChamber) {
                track.addRechit(rechit2d);
              } else {
                // add rechit to tree
                rechits2D_Chamber.push_back(chamber);
                rechits2D_X.push_back(rechit2d.getGlobalX());
                rechits2D_Y.push_back(rechit2d.getGlobalY());
                rechits2D_X_ClusterSize.push_back(rechit2d.getClusterSizeX());
                rechits2D_Y_ClusterSize.push_back(rechit2d.getClusterSizeY());
                rechits2D_X_Error.push_back(rechit2d.getErrorX());
                rechits2D_Y_Error.push_back(rechit2d.getErrorY());
              }
            }
            // fit and save track:
            track.fit();
            tracks_X_chi2.push_back(track.getChi2X());
            tracks_Y_chi2.push_back(track.getChi2Y());
            tracks_X_slope.push_back(track.getSlopeX());
            tracks_Y_slope.push_back(track.getSlopeY());
            tracks_X_intercept.push_back(track.getInterceptX());
            tracks_Y_intercept.push_back(track.getInterceptY());
            tracks_X_covariance.push_back(track.getCovarianceX());
            tracks_Y_covariance.push_back(track.getCovarianceY());

            // propagate to chamber under test:
            prophits2D_X.push_back(track.propagateX(detectorsTracker[testedChamber].getPositionZ()));
            prophits2D_Y.push_back(track.propagateY(detectorsTracker[testedChamber].getPositionZ()));
            prophits2D_X_Error.push_back(track.propagationErrorX(detectorsTracker[testedChamber].getPositionZ()));
            prophits2D_Y_Error.push_back(track.propagationErrorY(detectorsTracker[testedChamber].getPositionZ()));

            if (verbose) {
              std::cout << "  Chamber " << testedChamber << std::endl;
              std::cout << "    " << "track slope (" << track.getSlopeX() << "," << track.getSlopeY() << ")";
              std::cout << " " << "intercept (" << track.getInterceptX() << "," << track.getInterceptY() << ")";
              std::cout << std::endl;
              std::cout << "    " << rechits2D_X.size() << " rechits" << std::endl;
              if (rechits2D_X.size()>0) {
                  std::cout << "    " << "rechit (" << rechits2D_X.back();
                  std::cout << ", " << rechits2D_Y.back() << ")";
                  std::cout << "  " << "prophit (" << prophits2D_X.back();
                  std::cout << ", " << prophits2D_Y.back() << ")";
                  std::cout << std::endl;
              }
            }
          }
      } else {
        if (verbose) {
          std::cout << "  Not nice, skipping tracker calibration..." << std::endl; 
        }
      }

      if (verbose) {
          std::cout << "  #### Extrapolation on large detectors ####" << std::endl;
      }
      /* Build track with all trackers */
      trackerTracks.clear();
      allChi2.clear();
      
      /* Create all possible tracks,
       * then keep only the track with the lowest chi squared: */
 
      // Create unique array of chambers:
      std::set<int> chambersUniqueSet(vecRechit2DChamber->begin(), vecRechit2DChamber->end());
      std::vector<int> chambersUnique(chambersUniqueSet.begin(), chambersUniqueSet.end());
      int nChambersInEvent = chambersUnique.size();
      if (nChambersInEvent < 3) continue;
      if (verbose) {
          std::cout << "  There are " << nChambersInEvent << " trackers in the event: [";
          for (auto c:chambersUnique) std::cout << " " << c;
          std::cout << " ]" << std::endl;
      }

      // Divide the rechit indices in one vector per chamber:
      std::vector<std::vector<int>> rechitIndicesPerChamber(nChambersInEvent);
      for (int i=0; i<vecRechit2DChamber->size(); i++) {
          chamber = vecRechit2DChamber->at(i);
          int chamberIndex = std::find(chambersUnique.begin(), chambersUnique.end(), chamber) - chambersUnique.begin();
          rechitIndicesPerChamber[chamberIndex].push_back(i);
      }
      if (verbose) {
          for (int i=0; i<rechitIndicesPerChamber.size(); i++) {
              std::cout << "    Chamber " << chambersUnique[i] << ": ";
              for (auto rechitIndex:rechitIndicesPerChamber[i]) std::cout << rechitIndex << " ";
              std::cout << std::endl;
          }
      }
      // Create an array with an iterator for each chamber:
      std::vector<std::vector<int>::iterator> iterators;
      for (auto it=rechitIndicesPerChamber.begin(); it!=rechitIndicesPerChamber.end(); it++) {
          iterators.push_back(it->begin());
      }

      // Skip event if too many spurious hits:
      int nPossibleTracks = 1;
      for (auto el:rechitIndicesPerChamber) nPossibleTracks *= el.size();
      if (nPossibleTracks > 50) continue;
      if (verbose) {
          std::cout << "    There are " << nPossibleTracks << " possible tracks in the event..." << std::endl;
      }

      /* Create all possible rechit combinations per tracker
       * using "odometer" method:
       * https://stackoverflow.com/a/1703575
       */
      while (iterators[0] != rechitIndicesPerChamber[0].end()) {
          // build the track with current rechit combination:
          //std::this_thread::sleep_for(std::chrono::milliseconds(50));
          Track2D testTrack;
          for (auto it:iterators) {
              int rechitIndex = *it;
              chamber = vecRechit2DChamber->at(rechitIndex);
              rechit2d = Rechit2D(chamber,
                  Rechit(chamber, vecRechit2D_X_Center->at(rechitIndex), vecRechit2D_X_Error->at(rechitIndex), vecRechit2D_X_ClusterSize->at(rechitIndex)),
                  Rechit(chamber, vecRechit2D_Y_Center->at(rechitIndex), vecRechit2D_Y_Error->at(rechitIndex), vecRechit2D_Y_ClusterSize->at(rechitIndex))
              );
              detectorsTracker[chamber].mapRechit2D(&rechit2d); // apply local geometry
              testTrack.addRechit(rechit2d);
          }
          // build track and append it to the list:
          testTrack.fit();
          if (!testTrack.isValid()) {
            trackChi2X = -1.;
            trackChi2Y = -1.;
          } else {
            trackChi2X = testTrack.getChi2ReducedX();
            trackChi2Y = testTrack.getChi2ReducedY();
          }
          trackerTracks.push_back(testTrack);
          allChi2.push_back(trackChi2X + trackChi2Y);
          if (verbose) {
              std::cout << "    Built track with rechit IDs: ";
              for (auto it:iterators) std::cout << *it << " ";
              std::cout << " and chi2x " << trackChi2X << ", chi2y " << trackChi2Y << std::endl;
          }
          
          iterators[nChambersInEvent-1]++; // always scan the least significant vector
          for (int iChamber=nChambersInEvent-1; (iChamber>0) && (iterators[iChamber]==rechitIndicesPerChamber[iChamber].end()); iChamber--) {
              // if a vector arrived at the end, restart from the beginning
              // and increment the vector one level higher:
              iterators[iChamber] = rechitIndicesPerChamber[iChamber].begin();
              iterators[iChamber-1]++;
          }
      }
      int bestTrackIndex = 0;
      double bestTrackChi2 = 999;
      double presentTrackChi2;
      for (int i=0; i<trackerTracks.size(); i++) {
          //presentTrackChi2 = trackerTracks.at(i).getChi2ReducedX()+trackerTracks.at(i).getChi2ReducedY();
          presentTrackChi2 = allChi2.at(i);
          if (presentTrackChi2<bestTrackChi2) {
              bestTrackIndex = i;
              bestTrackChi2 = presentTrackChi2;
          }
      }
      track = trackerTracks.at(bestTrackIndex);
      allChi2.erase(allChi2.begin() + bestTrackIndex);
      if (verbose) {
          std::cout << "    Found best track at index " << bestTrackIndex;
          std::cout << " with chi2x " << track.getChi2ReducedX();
          std::cout << " and chi2y " << track.getChi2ReducedY() << ". ";
          std::cout << "Slope x " << track.getSlopeX() << ", intercept x " << track.getInterceptX() << ", ";
          std::cout << "slope y " << track.getSlopeY() << ", intercept y " << track.getInterceptY();
          std::cout << std::endl;
      }

      if (!track.isValid()) {
        trackChi2X = -1.;
        trackChi2Y = -1.;
      } else {
        trackChi2X = track.getChi2ReducedX();
        trackChi2Y = track.getChi2ReducedY();
        trackCovarianceX = track.getCovarianceX();
        trackCovarianceY = track.getCovarianceY();
        trackSlopeX = track.getSlopeX();
        trackSlopeY = track.getSlopeY();
        trackInterceptX = track.getInterceptX();
        trackInterceptY = track.getInterceptY();

        // extrapolate track on large detectors
        for (auto detector:detectorsLarge) {
          hit = track.propagate(&detector);
          prophitsChamber.push_back(detector.getChamber());
          prophitsEta.push_back(hit.getEta());
          prophitsGlobalX.push_back(hit.getGlobalX());
          prophitsGlobalY.push_back(hit.getGlobalY());
          prophitsErrorX.push_back(hit.getErrX());
          prophitsErrorY.push_back(hit.getErrY());
          prophitsLocalX.push_back(hit.getLocalX());
          prophitsLocalY.push_back(hit.getLocalY());
          prophitsLocalR.push_back(hit.getLocalR());
          prophitsLocalPhi.push_back(hit.getLocalPhi());

          if (verbose) {
            std::cout << "  Chamber " << detector.getChamber() << std::endl;
            std::cout << "    " << "track slope (" << track.getSlopeX() << "," << track.getSlopeY() << ")";
            std::cout << " " << "intercept (" << track.getInterceptX() << "," << track.getInterceptY() << ")";
            std::cout << std::endl;
            std::cout << "    " << "prophit " << "eta=" << hit.getEta() << ", ";
            std::cout << "global carthesian (" << hit.getGlobalX() << "," << hit.getGlobalY() << "), ";
            std::cout << "local carthesian (" << hit.getLocalX() << "," << hit.getLocalY() << "), ";
            std::cout << "local polar R=" << hit.getLocalR() << ", phi=" << hit.getLocalPhi();
            std::cout << std::endl;
          }
        }
      }

      // save all 1D rechits local coordinates
      for (int iRechit=0; iRechit<vecRechitChamber->size(); iRechit++) {
        chamber = vecRechitChamber->at(iRechit);
        if (verbose) std::cout << "  Chamber " << chamber << std::endl;
        if (detectorsMap.count(chamber)>0) {
          hit = Hit::fromLocal(detectorsMap.at(chamber),
            vecRechitX->at(iRechit), vecRechitY->at(iRechit), 0., 0., 0.
          );
        } else {
          if (verbose) std::cout << "    Skipping, no mapping found" << std::endl;
          continue;
        }
        rechitsChamber.push_back(chamber);
        rechitsEta.push_back(hit.getEta());
        rechitsLocalX.push_back(hit.getLocalX());
        rechitsLocalY.push_back(hit.getLocalY());
        rechitsLocalR.push_back(hit.getLocalR());
        rechitsLocalPhi.push_back(hit.getLocalPhi());
        rechitsGlobalX.push_back(hit.getGlobalX());
        rechitsGlobalY.push_back(hit.getGlobalY());
        rechitsClusterSize.push_back(vecRechitClusterSize->at(iRechit));
        if (verbose) {
          std::cout << "    " << "rechit  " << "eta=" << vecRechitEta->at(iRechit) << ", ";
          std::cout << "global carthesian (" << rechitsGlobalX.back() << "," << rechitsGlobalY.back() << "), ";
          std::cout << "local carthesian (" << hit.getLocalX() << "," << hit.getLocalY() << "), ";
          std::cout << "local polar R=" << hit.getLocalR() << ", phi=" << hit.getLocalPhi();
          std::cout << std::endl;
        }
      }

      trackTree.Fill();
    }
    std::cout << std::endl;
    std::cout << "Nice entries " << nentriesNice << std::endl;

    trackTree.Write();
    trackFile.Close();
    std::cout << "Output files written to " << ofile << std::endl;
}
