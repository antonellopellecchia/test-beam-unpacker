#include <cstdio>
#include <iostream>
#include <cstdint>
#include <vector>
#include <array>
#include <utility>
#include <bitset>
//#include "GEMAMC13EventFormat.h"
#include <TFile.h>
#include <TTree.h>

#include "StripMapping.h"
#include "ChamberMapping.h"
#include "GEMAMCEventFormat.h"

class GEMUnpacker {
  public:
    GEMUnpacker(const std::vector<std::string> ifilenames, const std::string isFedKit, const std::string _ofilename, const int _every) {
      try {
        for (auto ifilename:ifilenames)
          m_files.push_back(std::fopen(ifilename.c_str(), "rb"));
      } catch (int e) {
        std::cout << "An exception occured. Exception code " << e << std::endl;
      }
      ofilename = _ofilename;
      m_isFedKit = isFedKit;
      every = _every;
    }

    ~GEMUnpacker() {
      for (auto file:m_files)
        if (file != NULL) std::fclose(file);
    }

    void setParameters(bool _verbose, bool _checkSyncronization) {
      verbose = _verbose;
      checkSyncronization = _checkSyncronization;
    }

    int readEvent(int slot) {
      // slot = _slot;

      // read and print FEROL headers
      if (m_isFedKit == "ferol") {
        std::size_t sz = std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
        if (sz == 0 ) return -1; // end of file reached
        //printf("%016llX\n", m_word);
        std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
        //printf("%016llX\n", m_word);
        std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
        //printf("%016llX\n", m_word);
        // ferol headers read and printed, now read CDF header
        //std::fread(&m_word, sizeof(uint64_t), 1, m_file);
      } else {
        std::size_t sz = std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
        if (sz == 0 ) return -1; // end of file reached
        //read and print "BADC0FFEEBADCAFE" and another artificial header
        //printf("%016llX\n", m_word);
        //std::fread(&m_word, sizeof(uint64_t), 1, m_file);
        //printf("%016llX\n", m_word);
      }

      // m_AMC13Event = new AMC13Event();
      // //std::fread(&m_word, sizeof(uint64_t), 1, m_file);
      // //printf("%016llX\n", m_word);
      // m_AMC13Event->setCDFHeader(m_word);
      // std::fread(&m_word, sizeof(uint64_t), 1, m_file);
      // //printf("%016llX\n", m_word);
      // m_AMC13Event->setAMC13header(m_word);
      // //printf("%016llX\n", m_word);
      // //std::cout << "n_AMC = " << m_AMC13Event->nAMC() << std::endl;
      // // Readout out AMC headers
      // for (unsigned short i = 0; i < m_AMC13Event->nAMC(); i++){
      //     std::fread(&m_word, sizeof(uint64_t), 1, m_file);
      //     //printf("%016llX\n", m_word);
      //     m_AMC13Event->addAMCheader(m_word);
      // }

      // Readout out AMC payloads
      AMCEvent * m_amcEvent = new AMCEvent();
      vfatId = 0;
      oh = 0;

      std::fread(&m_word, 8, 1, m_files.at(slot));
      //printf("AMC HEADER1\n");
      //printf("%016llX\n", m_word);
      m_amcEvent->setAMCheader1(m_word);
      std::fread(&m_word, 8, 1, m_files.at(slot));
      //printf("AMC HEADER2\n");
      //printf("%016llX\n", m_word);
      m_amcEvent->setAMCheader2(m_word);
      std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
      m_amcEvent->setGEMeventHeader(m_word);

      l1a = m_amcEvent->EC();
      if (checkSyncronization) {
        for (int otherL1A:eventVecL1A) {
          if (l1a!=otherL1A) {
            if (verbose) {
              std::cout << "ERROR: different L1As in event: ";
              std::cout << l1a << " " << otherL1A << ", exiting..." << std::endl;
            }
            return 128; // l1a out of sync
          }
        }
        eventVecL1A.push_back(l1a);
      }
      runParameter = m_amcEvent->RunParameter();
      //std::cout << "Run parameter " << runParameter << std::endl;
      pulse_stretch = m_amcEvent->PULSE_STRETCH();
      orbitNumber = m_amcEvent->Onum();
      bunchCounter = m_amcEvent->BC();
      eventCounter = m_amcEvent->EC();

      if (verbose) {
          std::cout << "    Run type " << (int) m_amcEvent->Rtype() << ", ";
          std::cout << "BC " << m_amcEvent->BC() << ", ";
          std::cout << "EC " << m_amcEvent->EC() << std::endl;
      }

      //printf("GEM EVENT HEADER\n");
      //printf("%013llX\n", m_word);
      // fill the geb data here
      //std::cout << "GDcount = " << m_amcEvent->GDcount() << std::endl;
      for (unsigned short j = 0; j < m_amcEvent->GDcount(); j++) {
          GEBdata * m_gebdata = new GEBdata();
          std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
          m_gebdata->setChamberHeader(m_word);
          //printf("GEM CHAMBER HEADER\n");
          //printf("%016llX\n", m_word);
          // fill the vfat data here
          //std::cout << "Number of VFAT words " << m_gebdata->Vwh() << std::endl;
          //std::fread(&m_word, sizeof(uint64_t), 1, m_file);
          int m_nvb = m_gebdata->Vwh() / 3; // number of VFAT2 blocks. Eventually add here sanity check
          //printf("N vfat blocks %d\n",m_nvb);
          //printf("OH %d\n",m_gebdata->InputID());
          
          oh = m_gebdata->InputID();
          //if (oh==1) continue;
          
          if (verbose) {
              std::cout << "        " << "slot\toh\tvfat\tBC\tEC\teta\tchamber\tcrc\tdata" << std::endl;
          }
          for (unsigned short k = 0; k < m_nvb; k++) {
            VFATdata * m_vfatdata = new VFATdata();
            // read 3 vfat block words, totaly 192 bits
            std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
            //printf("VFAT WORD 1 ");
            //std::cout << std::bitset<64>(m_word) << std::endl;
            //printf("%016llX\n", m_word);
            //printf("%016llX\n", m_word >> 56);
            //printf("%016llX\n", 0x3f);
            //printf("%016llX\n", 0x3f & (m_word >> 56));
            m_vfatdata->read_fw(m_word);
            std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
            //printf("VFAT WORD 2 ");
            //std::cout << std::bitset<64>(m_word) << std::endl;
            //printf("%016llX\n", m_word);
            m_vfatdata->read_sw(m_word);
            std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
            //printf("VFAT WORD 3 ");
            //std::cout << std::bitset<64>(m_word) << std::endl;
            //printf("%016llX\n", m_word);
            m_vfatdata->read_tw(m_word);

            vfatId = m_vfatdata->Pos();

            crc = m_vfatdata->CRCcheck();
            if (crc>0) {
                if (verbose) {
                    std::cout << "\tCRC error " << crc << " in VFAT " << vfatId << ", skipping..." << std::endl;
                }
                continue;
            }

            chamber = chamberMapping->to_chamber[slot][oh][vfatId];
            if (stripMappings.count(chamber)==0) continue;
            StripMapping *stripMapping = stripMappings.at(chamber);
            eta = stripMapping->to_eta[vfatId];


            if (verbose) {
              std::cout << "        " << slot << "\t" << oh << "\t" << std::setw(3) << vfatId;
              std::cout << "\t" << (int) m_vfatdata->BC() << "\t" << (int) m_vfatdata->EC();
              std::cout << "\t" << eta << "\t" << chamber; 
              std::cout << "\t" << (int) m_vfatdata->CRCcheck();
              std::cout << "\t" << std::bitset<64>(m_vfatdata->msData());
              std::cout << "" << std::bitset<64>(m_vfatdata->lsData());
              std::cout << std::endl;
            }

            direction = eta%2;
            for (int i=0;i<64;i++) {
              if (m_vfatdata->lsData() & (1LL << i)) {
                vecCh.push_back(i);
                vecVfat.push_back(vfatId);
                vecOh.push_back(oh);
                vecSlot.push_back(slot);
                vecDigiEta.push_back(eta);
                vecDigiChamber.push_back(chamber);
                vecDigiDirection.push_back(direction);
                vecDigiStrip.push_back(stripMapping->to_strip[vfatId][i]);
                nhits++;
              }
              if (m_vfatdata->msData() & (1LL << i)) {
                vecCh.push_back(i+64);
                vecVfat.push_back(vfatId);
                vecOh.push_back(oh);
                vecSlot.push_back(slot);
                vecDigiEta.push_back(eta);
                vecDigiChamber.push_back(chamber);
                vecDigiDirection.push_back(direction);
                vecDigiStrip.push_back(stripMapping->to_strip[vfatId][i+64]);
                nhits++;
              }
            }
            delete m_vfatdata;
          }
          std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
          m_gebdata->setChamberTrailer(m_word);
          m_amcEvent->g_add(*m_gebdata);
          delete m_gebdata;
      }
      std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
      m_amcEvent->setGEMeventTrailer(m_word);
      std::fread(&m_word, sizeof(uint64_t), 1, m_files.at(slot));
      //printf("AMC TRALIER\n");
      //printf("%016llX\n", m_word);
      m_amcEvent->setAMCTrailer(m_word);

      delete m_amcEvent;
      return 0;
    }    

    int unpack(const int max_events, std::map<int, StripMapping*> _stripMappings, ChamberMapping* _chamberMapping) {
      stripMappings = _stripMappings;
      chamberMapping = _chamberMapping;

      int unpackerStatus = 0;

      if (max_events > 0) std::cout << "Unpacking " << max_events << " events" << std::endl;
      else std::cout << "Unpacking all events" << std::endl; 

      TFile *hfile = new TFile(ofilename.c_str(), "RECREATE", "GEM Raw ROOT");
      TTree outputtree("outputtree", "outputtree");

      outputtree.Branch("orbitNumber", &orbitNumber);
      outputtree.Branch("bunchCounter", &bunchCounter);
      outputtree.Branch("eventCounter", &eventCounter);
      outputtree.Branch("nhits", &nhits);
      outputtree.Branch("runParameter", &runParameter);
      outputtree.Branch("pulse_stretch", &pulse_stretch);
      outputtree.Branch("slot", &vecSlot);
      outputtree.Branch("OH", &vecOh);
      outputtree.Branch("VFAT", &vecVfat);
      outputtree.Branch("CH", &vecCh);

      // digi variable branches
      outputtree.Branch("digiChamber", &vecDigiChamber);
      outputtree.Branch("digiEta", &vecDigiEta);
      outputtree.Branch("digiDirection", &vecDigiDirection);
      outputtree.Branch("digiStrip", &vecDigiStrip);

      int n_evt=0;

      while (true) {
        if ((max_events>0) && (n_evt>max_events)) break;
        if ( (!verbose) && n_evt%1000==0 ) std::cout << "Unpacking event " << n_evt << "         \r";

        // reset all branch variables:
        nhits=0;
        vecSlot.clear();
        vecOh.clear();
        vecVfat.clear();
        vecCh.clear();

        vecDigiEta.clear();
        vecDigiChamber.clear();
        vecDigiDirection.clear();
        vecDigiStrip.clear();

        eventVecL1A.clear();
        // read event from raw:
        if (verbose) std::cout << "Event " << n_evt << std::endl;
        for (int slot=0; slot<m_files.size(); slot++) {
          // slot == file index. To be improved?
          if (verbose) {
            std::cout << "    File " << m_files.at(slot) << std::endl;
          }
          readStatus = readEvent(slot);
        }
        if (readStatus<0) break; // end of file
        else if (readStatus==128) {
          unpackerStatus = readStatus; // L1A out of sync
          std::cout << "Found mismatching L1As in event " << n_evt << ", stopping..." << std::endl;
          break;
        }

        if (every==0 || n_evt%every==0) { // save only every n events
            outputtree.Fill();
            //std::cout << std::endl << "Saving event " << n_evt << ": n_evt/every = " << n_evt%every << std::endl;
        }
        n_evt++; 
      }
      std::cout << std::endl;
      hfile->Write();
      return unpackerStatus;
    }

private:
    int readStatus = 0;
    
    /* branch and support variables: */
    int nhits;
    int runParameter;
    int pulse_stretch;
    int orbitNumber, bunchCounter, eventCounter;
    int l1a;
    std::vector<int> eventVecL1A;
    // raw variables
    std::vector<int> vecSlot;
    std::vector<int> vecOh;
    std::vector<int> vecVfat;
    std::vector<int> vecCh;
    // digi variables
    std::vector<int> vecDigiEta; // even for x, odd for y
    std::vector<int> vecDigiChamber; // 0 to 3 for trackers, 4 for GE21, 5 for ME0
    std::vector<int> vecDigiDirection; // 0 for x, 1 for y
    std::vector<int> vecDigiStrip; // 0 to 357
    // support variables
    int vfatId = 0;
    //int slot = 0;
    int oh = 0;
    int eta = 0;
    int strip = 0;
    int chamber = 0;
    int direction = 0;
    int crc = 0;

    /* raw data variables: */
    std::vector<std::FILE *> m_files;
    uint64_t m_word;
    uint32_t m_word32;
    uint64_t fw_;
    uint64_t sw_;
    uint64_t tw_;
    bool type;
    AMCEvent * m_AMCEvent;
    std::string ofilename;
    std::string m_isFedKit;
    int every; // events to skip

    std::map<int, StripMapping*> stripMappings;
    ChamberMapping *chamberMapping;
    bool verbose=false, checkSyncronization=false;
};
 
int main (int argc, char** argv) {
  std::cout << "Running GEM unpacker..." << std::endl;
  if (argc<3) 
  {
    std::cout << "Usage: RawToDigi ifile(s) ofile [--events max_events] [--geometry geometryname] [--format ferol/sdram] [--verbose] [--check-sync] --every [events]" << std::endl;
    return 0;
  }
  std::vector<std::string> ifiles;
  std::string ofile;
  std::string isFedKit = "ferol";
  
  int max_events = -1;
  int every = 0;
  std::string geometry = "oct2021";
  bool verbose = false;
  bool checkSyncronization = false;
  bool isUnnamed = true;
  for (int iarg=1; iarg<argc; iarg++) {
    std::string arg = argv[iarg];
    if (arg[0]=='-') { // parse named parameters
      isUnnamed = false; // end of unnamed parameters
      if (arg=="--verbose") verbose = true;
      else if (arg=="--check-sync") checkSyncronization = true;
      else if (arg=="--events") max_events = atoi(argv[iarg+1]);
      else if (arg=="--geometry") geometry = argv[iarg+1];
      else if (arg=="--every") every = atoi(argv[iarg+1]);
      std::cout << "every " << every << std::endl;
    } else if (isUnnamed) { // unnamed parameters
      if (iarg+1==argc || argv[iarg+1][0]=='-') ofile = arg;
      else ifiles.push_back(arg);
    }
  }
  std::cout << "ifiles ";
  for (auto s:ifiles) std::cout << s << " ";
  std::cout << std::endl;
  std::cout << "ofile " << ofile << std::endl;

  std::cout << "Reading mapping files..." << std::endl;
  std::string mappingBaseDir = "mapping/"+geometry;
  std::map<int, StripMapping*> stripMappings;
  ChamberMapping chamberMapping(mappingBaseDir+"/chamber_mapping.csv");

  if (geometry=="oct2021" || geometry=="may2022") {
      StripMapping trackerStripMapping(mappingBaseDir+"/tracker_mapping.csv");
      StripMapping ge21StripMapping(mappingBaseDir+"/ge21_mapping.csv");
      StripMapping me0StripMapping(mappingBaseDir+"/me0_mapping.csv");
      std::cout << "Mapping files ok." << std::endl;

      stripMappings = {
        {0, &trackerStripMapping},
        {1, &trackerStripMapping},
        {2, &trackerStripMapping},
        {3, &trackerStripMapping},
        {4, &ge21StripMapping},
        {5, &me0StripMapping},
        {6, &me0StripMapping},
      };
  } else if (geometry=="july2022") {
      StripMapping trackerStripMapping(mappingBaseDir+"/tracker_mapping.csv");
      StripMapping me0StripMapping(mappingBaseDir+"/me0_mapping.csv");
      std::cout << "Mapping files ok." << std::endl;

      stripMappings = {
        {0, &trackerStripMapping},
        {1, &trackerStripMapping},
        {2, &trackerStripMapping},
        {3, &me0StripMapping},
      };
  } else {
      std::cout << "Error: geometry " << geometry << " not supported yet." << std::endl;
      return -1;
  }

  if (verbose) {
      chamberMapping.print();
      /*for (auto stripMapping:stripMappings) {
          std::cout << "Chamber " << stripMapping.first << std::endl;
          stripMapping.second->print();
      }*/
  }

  GEMUnpacker * m_unpacker = new GEMUnpacker(ifiles, isFedKit, ofile, every);
  m_unpacker->setParameters(verbose, checkSyncronization);
  int unpackerStatus = m_unpacker->unpack(max_events, stripMappings, &chamberMapping);
  delete m_unpacker;
  std::cout << "Output file saved to " << ofile << std::endl;

  return unpackerStatus;
}
