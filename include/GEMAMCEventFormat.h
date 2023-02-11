#include <cstdio>
#include <iostream>
#include <vector>
//!A class for VFAT data
class VFATdata 
{
  private:
    // Currently used VFAT3 lossless data format. The size of one VFAT data block is 192 bits. Bits[175:0] come from the VFAT3, and bits[191:176] are appended by the CTP7.
    uint8_t  fPos;                     ///<an 8bit value indicating the VFAT position on this GEB (it can be 0 to 23, so really only 6 bits are used, but anyway...)
    uint8_t  fCRCcheck;                ///<bits 183:177 are not used, should be 0, bit 176 is 1 if CTP7 detected a CRC mismatch
   uint8_t  fHeader;                  /// Either 0x5E (normally it should be 0x1E, if it is 0x5E that indicates that the VFAT3 internal buffer is half-full, so it's like a warning)
    uint8_t  fEC;                      ///<Event Counter, 8 bits
    uint16_t  fBC;                     ///<Bunch counter
    uint64_t flsData;                  ///<channels from 1to64 
    uint64_t fmsData;                  ///<channels from 65to128
    uint16_t fcrc;                     ///<Check Sum value, 16 bits
    uint16_t fcrc_calc;                ///<Check Sum value recalculated, 16 bits
    int      fSlotNumber;              ///<Calculated chip position
    bool     fisBlockGood;             ///<Shows if block is good (control bits, chip ID and CRC checks)
  
  public:

  union VFATfirst {
      uint64_t word;
      // v3 dataformat
      struct {
        uint64_t msData1 : 16;  ///<channels from 65to128
        uint64_t bc : 16;       ///<Bunch Crossing number, 16 bits
        uint64_t ec : 8;        ///<Event Counter, 8 bits
        uint64_t header : 8;    ///<normally 0x1E. 0x5E indicates that the VFAT3 internal buffer
        // is half-full, so it's like a warning
        uint64_t vc : 1;   /// VFAT CRC Error
        uint64_t : 7;      // unused
        uint64_t pos : 8;  // VFAT position on chamber, 5 used in GE11 but more is needed for phase2
      };
      // v2 dataformat
      struct {
        uint64_t msData1v2 : 16;  ///<channels from 65to128 - placeholder since msData1 reads same info
        uint64_t chipID : 12;     ///<Chip ID, 12 bits
        uint64_t b1110 : 4;       ///<1110:4 Control bits, shoud be 1110
        uint64_t flag : 4;        ///<Control Flags: 4 bits, Hamming Error/AFULL/SEUlogic/SUEI2C
        uint64_t ecV2 : 8;        ///<Event Counter, 8 bits
        uint64_t b1100 : 4;       ///<1100:4, Control bits, shoud be 1100
        uint64_t bcV2 : 12;       ///<Bunch Crossing number, 12 bits
        uint64_t b1010 : 4;       ///<1010:4 Control bits, shoud be 1010
      };
    };
    union VFATsecond {
      uint64_t word;
      struct {
        uint64_t lsData1 : 16;  ///<channels from 1to64
        uint64_t msData2 : 48;  ///<channels from 65to128
      };
    };
    union VFATthird {
      uint64_t word;
      struct {
        uint64_t crc : 16;      ///<Check Sum value, 16 bits
        uint64_t lsData2 : 48;  ///<channels from 1to64
      };
    };

    //!Empty constructor. Functions used to assign data members.
    VFATdata(){}
    //!Constructor requiring arguments.
    VFATdata(const uint8_t &Pos_, 
            const uint8_t &CRCcheck_,
            const uint8_t &Header_,
            const uint8_t &EC_,
            const uint16_t &BC_,
            const uint64_t &lsData_, 
            const uint64_t &msData_, 
            const uint16_t &crc_,
            const uint16_t &crc_calc_,
            const int &SlotNumber_,
            const bool &isBlockGood_) : 
        fPos(Pos_),
        fCRCcheck(CRCcheck_),
        fHeader(Header_),
        fEC(EC_),
        fBC(BC_),
        flsData(lsData_),
        fmsData(msData_),
        fcrc(crc_),
        fcrc_calc(crc_calc_),
        fSlotNumber(SlotNumber_),
        fisBlockGood(isBlockGood_){}
    //!Destructor for the class
    ~VFATdata(){}

    //!Read first word from the block.
    void read_fw(uint64_t word)
    {
      fPos = (int)(0x3f & (word >> 56));
      //std::cout << (0x3f & (word >> 56)) << std::endl;
      //std::cout << fPos << std::endl;
      fCRCcheck = 0xff & (word >> 48);
      fHeader = 0xff & (word >> 40);
      fEC = (uint16_t) (0xff & (word >> 32));
      fBC = 0xffff & (word >> 16);
      fmsData = 0xffff000000000000 & (word << 48);
    }
    
    //!Read second word from the block.
    void read_sw(uint64_t word)
    {
      fmsData = fmsData | (0x0000ffffffffffff & word >> 16);
      flsData = 0xffff000000000000 & (word << 48);
    }
    
    //!Read third word from the block.
    void read_tw(uint64_t word)
    {
      flsData = flsData | (0x0000ffffffffffff & word >> 16);
      fcrc = 0xffff & word;
    }
    
    uint16_t   Pos        (){ return fPos;        }
    uint16_t  BC         (){ return fBC;         }
    uint8_t   Header     (){ return fHeader;     }
    uint8_t   EC         (){ return fEC;         }
    uint8_t   CRCcheck   (){ return fCRCcheck;   }
    uint64_t  lsData     (){ return flsData;     }
    uint64_t  msData     (){ return fmsData;     }
    uint16_t  crc        (){ return fcrc;        }
    uint16_t  crc_calc   (){ return fcrc_calc;   }
    int       SlotNumber (){ return fSlotNumber; }
    bool      isBlockGood(){ return fisBlockGood;}

};

//!A class for GEB data
/*!
 The number after the ":" indicates how many bits a certain item consists of. 
*/
class GEBdata
{
  private:
    std::vector<VFATdata> vfatd;     ///<Vector of VFAT data
    std::vector<uint8_t> v_GEBflags; ///<Vector for thirteen flags in GEM Chamber Header

    //GEM chamber header

      //!Zero Suppression Flags:24  (8 zeroes):8
      /*!Bitmask indicating if certain VFAT blocks have been zero suppressed*/
      uint32_t m_ZeroSup;
      //!Input ID:5    000:3
      /*!GLIB input ID (starting at 0)*/
      uint8_t m_InputID;   
      //!VFAT word count:12   0000:4
      /*!Size of VFAT payload in 64 bit words*/
      uint16_t m_Vwh;
      //!Thirteen Flags, only one bit each
      /*! 
       000:3    EvtFIFO full:1    InFIFO full:1   L1AFIFO full:1    Even size overflow:1    EvtFIFO near full:1   InFIFO near full:1    
       L1AFIFO near full:1    Event size warn:1   No VFAT marker:1    OOS GLIB VFAT:1   OOS GLIB OH:1   
       BX mismatch GLIB VFAT:1    BX mismatch GLIB OH:1
      */
      uint16_t m_ErrorC;

    //GEM chamber trailer
		
      //!OH CRC:16
      /*!CRC of OH data (currently not available)*/
      uint16_t m_OHCRC;     
      //!0000:4   VFAT word count:12   
      /*!Same as in header. This one actually counts the number of valid words that were sent to AMC13; the one in header is what we expected to send to AMC13*/
      uint16_t m_Vwt;      
      //!(7 0's):7    InFIFO underflow:1   
      /*!Input status (critical): Input FIFO underflow occured while sending this event*/
      uint8_t m_InFu;    
      //!(7 0's):7    Stuck data:1    
      /*!Input status (warning): Data in InFIFO or EvtFIFO when L1A FIFO was empty. Only resets with resync or reset*/
		  uint8_t m_Stuckd; 

  public:
    //!Empty constructor. Functions used to assign data members.
    GEBdata(){};
    //!Constrcutor requirng arguments.
    GEBdata(const uint32_t &ZeroSup_, 
              const uint8_t &InputID_, 
              const uint16_t &Vwh_, 
              const uint16_t &ErrorC_, 
              const uint16_t &OHCRC_, 
              const uint16_t &Vwt_,
              const uint8_t &InFu_,
              const uint8_t &Stuckd_) : 
          m_ZeroSup(ZeroSup_),                                
          m_InputID(InputID_),
          m_Vwh(Vwh_),                                  
          m_ErrorC(ErrorC_),
          m_OHCRC(OHCRC_),                                    
          m_Vwt(Vwt_),                             
          m_InFu(InFu_),                                   
          m_Stuckd(Stuckd_){}  
    //!Destructor for the class.
    ~GEBdata(){vfatd.clear();}

    // need to include all the flags
    //!Reads the word for the GEM Chamber Header. Puts the thirteen flags in a vector.
    /*!
     Fills the Zero Suppression, GLIB Input ID, VFAT word count, and Thirteen Flags.
     */
    void setChamberHeader(uint64_t word)
    {
      m_ZeroSup = 0x00ffffff & (word >> 40);        /*!Zero Suppression*/
      m_InputID = (int) (0b00011111 & (word >> 35));        /*!GLIB Input ID*/
      //std::cout << "chamber header " << word << std::endl; 
      //std::cout << "word " << word << std::endl; 
      m_Vwh = 0x0fff & (word >> 23);                /*!VFAT word count*/
      m_ErrorC = 0b0001111111111111 & (word >> 10);    /*!Thirteen Flags*/
      for(int i=0; i<13; ++i)
      {
        v_GEBflags.push_back(0x01 & (m_ErrorC >> i));
      }
    }

    //return specific flags
    //!Returns one of thirteen flags from GEM chamber header.
    /*!
     Argument must be between 0 and 12. The flags corresponding to a given argument are shown.
     12->EvtFIFO full    11->InFIFO full    10->L1AFIFO full   9->Even size overflow    8->EvtFIFO near full   5->InFIFO near full    
     6->L1AFIFO near full    5->Event size warn   4->No VFAT marker    3->OOS GLIB VFAT   2->OOS GLIB OH 
     1->BX mismatch GLIB VFAT    0->BX mismatch GLIB OH
    */
    uint8_t GEBflag(int c)
    {
      return v_GEBflags[c];
    }
    
    // need to include all the flags
    //!Reads the word for GEM Chamber Trailer
    /*!
     Fills the OH CRC, VFAT word count, InFIFO underflow, and Stuck data.
    */
    void setChamberTrailer(uint64_t word)
    {
      m_OHCRC = word >> 48;           /*!OH CRC*/
      m_Vwt = 0x0fff & (word >> 36);  /*!VFAT word count*/
      m_InFu = 0x01 & (word >> 35);   /*!InFIFO underflow*/
      m_Stuckd = 0x01 & (word >> 34); /*!Stuck data*/
    }

    uint32_t ZeroSup()  {return m_ZeroSup;}   ///<Returns Zero Suppression flags
    uint8_t  InputID()  {return m_InputID;}   ///<Returns GLIB input ID
    uint16_t Vwh()      {return m_Vwh;}       ///<Returns VFAT word count (size of VFAT payload)
    uint16_t ErrorC()   {return m_ErrorC;}    ///<Returns thirteen flags in GEM Chamber Header

    uint16_t OHCRC()    {return m_OHCRC;}     ///<Returns OH CRC 
    uint16_t Vwt()      {return m_Vwt;}       ///<Returns VFAT word count
    uint8_t  InFu()     {return m_InFu;}      ///<Returns InFIFO underflow flag
    uint8_t  Stuckd()   {return m_Stuckd;}    ///<Returns Stuck data flag

    //!Adds VFAT data to the vector
    void v_add(VFATdata v){vfatd.push_back(v);}
    //!Returns the vector of FVAT data
    std::vector<VFATdata> vfats(){return vfatd;}  
};

//!A class for AMC data
/*!
 The number after the ":" indicates how many bits a certain item consists of.
*/
class AMCEvent
{
  private:
    std::vector<GEBdata> gebd;    ///<Vector of GEB data

    //AMC header #1	

      //!0000:4   AMC#:4
      /*!Slot number of AMC(GLIB/MP7/EC7, etc.)*/
      uint8_t  m_AMCnum;        
      //!(8 zeroes):8    L1A ID:24    
      /*!Basically like event number, but reset by resync*/
      uint32_t m_EC;          
      //!0000:4   BX ID:12         
      /*!Bunch crossing ID*/
      uint16_t m_BC;    
      //!(12 zeroes):12    Data length:20   
      /*!Overall size of this FED event fragment in 64bit words (including headers and trailers)*/
      uint32_t m_Dlength;

    //AMC header #2

      uint8_t m_FV;           ///<0000:4    Format Version:4    
      //!0000:4   Run Type:4   
      /*!Current version = 0x0;  Could be used to encode run types like physics, cosmics, threshold scan, etc.*/
      uint8_t m_Rtype;
      //adapt for the moment to lat scan only, TODO: generalize. See https://github.com/cms-gem-daq-project/cmsgemos/issues/230
      uint8_t m_isExtTrig;
      uint8_t m_isCurrentPulse;
      uint8_t m_CFG_CAL_DAC;
      uint8_t m_PULSE_STRETCH;
      uint16_t m_RunParameter;
      //uint8_t m_Param1;       ///<Run param1:8 
      //uint8_t m_Param2;       ///<Run param2:8
      //uint16_t m_Param3;       ///<Run param3:8 //GC : messo a 16
      uint16_t m_Onum;        ///<Orbit number:16
      //!Board ID:16
      /*!This is currently filled with 8bit long GLIB serial number*/
      uint16_t m_BID;

    //GEM event header

      //!(8 zeroes):8   GEM DAV list:24    
      /*!Bitmask indicating which inputs/chambers have data*/
      uint32_t m_GEMDAV; 
      //!(30 zeroes):30    Buffer Status:34  
      /*!Bitmask indicating buffer error in given inputs*/
      uint64_t m_Bstatus;  
      //!000:3   GEM DAV count:5    
      /*!Number of chamber blocks*/
      uint8_t  m_GDcount;   
      //!0000:4    TTS state:4    
      /*!Debug: GLIB TTS state at the moment when this event was built*/
      uint8_t  m_Tstate;

    //GEM event trailer

      //!(8 zeroes):8    Chamber timeout:24   
      /*!Bitmask indicating if GLIB did not recieve data from particular input for this L1A in X amount of GTX clock cycles*/
      uint32_t m_ChamT;     
      //!(7 zeroes):7   OOS GLIB:1   
      /*!GLIB is out-of-sync (critical): L1A ID is different for different chambers in this event.*/
      uint8_t  m_OOSG;

    //AMC_trailer

      uint32_t m_CRC;
      uint8_t m_L1AT;
      uint32_t m_DlengthT;
	
  public:
    //!Empty constructor. Functions fill the data members.
    AMCEvent(){};
    //!Constructor requiring arguments.
    AMCEvent(const uint8_t &AMCnum_, 
              const uint32_t &EC_,
              const uint16_t &BC_, 
              const uint32_t &Dlength_,
              const uint8_t &FV_,
              const uint8_t &Rtype_, 
              const uint8_t &isExtTrig_, 
              const uint8_t &isCurrentPulse_, 
              const uint8_t &CFG_CAL_DAC_, 
              const uint8_t &PULSE_STRETCH_, 
              const uint16_t &RunParameter_,      
              const uint16_t &Onum_, 
              const uint16_t &BID_,
              const uint32_t &GEMDAV_, 
              const uint64_t &Bstatus_,
              const uint8_t &GDcount_, 
              const uint8_t &Tstate_,
              const uint32_t &ChamT_,
              const uint8_t OOSG_) :
          m_AMCnum(AMCnum_), 
          m_EC(EC_),
          m_BC(BC_),                                 
          m_Dlength(Dlength_),
          m_FV(FV_),
          m_Rtype(Rtype_),                                
          m_isExtTrig(isExtTrig_),
          m_isCurrentPulse(isCurrentPulse_),                                  
          m_CFG_CAL_DAC(CFG_CAL_DAC_),
          m_PULSE_STRETCH(PULSE_STRETCH_),
          m_RunParameter(RunParameter_),
          m_Onum(Onum_),                                    
          m_BID(BID_),
          m_GEMDAV(GEMDAV_), 
          m_Bstatus(Bstatus_),
          m_GDcount(GDcount_),                                 
          m_Tstate(Tstate_),                             
          m_ChamT(ChamT_),                                   
          m_OOSG(OOSG_){}
    //!Destructor for the class
    ~AMCEvent(){gebd.clear();}

    //!Reads the word for AMC Header 
    /*!
     Fills the AMC number, L1A ID, BX ID, and Data Length
    */
    void setAMCheader1(uint64_t word)
    {
      m_AMCnum = 0x0f & (word >> 56);     /*!AMC number*/
      m_EC = 0x00ffffff & (word >> 32);  /*!L1A ID */
      m_BC = 0x0fff & (word >> 20);       /*!BX ID */
      m_Dlength = 0x000fffff & word;      /*!Data Length */
      //std::cout << "L1A_ID " << m_L1A << std::endl;
      //std::cout << "BX " << m_BX << std::endl;
      //std::cout << "Data length " << m_Dlength << std::endl;
    }
    
    //!Reads the word for the AMC Header 2
    /*!
     Fills the Format Version, Run Type, Run Param 1, Run Param 2, Run Param 3, Orbit Number, and Board ID
    */
    void setAMCheader2(uint64_t word)
    {
      m_FV = 0x0f & (word >> 60);     /*!Format Version */
      m_Rtype = 0x0f & (word >> 56);  /*!Run Type */
      m_isExtTrig = 0x1 & (word >> 54);
      m_isCurrentPulse = 0x1 & (word >> 53);
      m_CFG_CAL_DAC = 0xff & (word >> 45);
      m_PULSE_STRETCH = (int) (0x07 & (word >> 42));/* 3 bit */
      //std::cout << "ps " << (0x07 & (word >> 42)) << std::endl;
      //m_RunParameter = 0x03ff & (word >> 32); //word >> 32;          /*!Run Param 3 */
      m_Onum = word >> 16;            /*!Orbit Number */
      m_BID = word;                   /*!Board ID */
      //std::cout << "OC " << m_Onum << std::endl;
    }
    
    //!Reads the word for the GEM Event Header
    /*!
     Fills the GEM DAV list, Buffer Status, GEM DAV count, and TTS state.
    */
    void setGEMeventHeader(uint64_t word)
    {
      m_GEMDAV = 0x00ffffff & (word >> 40);   /*!GEM DAV list*/
      m_Bstatus = 0x00ffffff & (word >> 16);  /*!Buffer Status*/
      m_GDcount = 0b00011111 & (word >> 11);  /*!GEM DAV count*/
      m_Tstate = 0b00000111 & word;           /*!TTS state*/
    }

    //!Reads the word for the GEM Event Trailer
    /*!
     Fills the Chamber Timeout and OOS GLIB.
    */
    void setGEMeventTrailer(uint64_t word)
    {
      m_ChamT = 0x00ffffff & (word >> 40);  /*!Chamber Timeout*/
      m_OOSG = 0b00000001 & (word >> 39);   /*!OOS GLIB*/
      m_RunParameter = 0x03ff & (word >> 8); 
      m_PULSE_STRETCH = (int) (0x07 & (word >> 18));
    }

    //!Reads the word for the AMC Trailer
    void setAMCTrailer(uint64_t word)
    {
      m_CRC = word >> 32;
      m_L1AT = word >> 24;
      m_DlengthT = 0x000fffff & word;
    }

    uint8_t  AMCnum()  {return m_AMCnum;}   ///<Returns AMC number
    uint32_t EC()     {return m_EC;}      ///<Returns L1A number
    uint16_t BC()      {return m_BC;}       ///<Returns Bunch Crossing ID
    uint32_t Dlength() {return m_Dlength;}  ///<Returns Data Length (Overall size of FED event fragment)

    uint8_t  FV()      {return m_FV;}       ///<Returns Format Version
    uint8_t  Rtype()   {return m_Rtype;}    ///<Returns Run Type
    uint8_t  isExtTrig()  {return m_isExtTrig;}   
    uint8_t  isCurrentPulse()  {return m_isCurrentPulse;}
    uint8_t  CFG_CAL_DAC()  {return m_CFG_CAL_DAC;}
    uint8_t  PULSE_STRETCH()  {return m_PULSE_STRETCH;}
    uint16_t RunParameter()  {return m_RunParameter;}
    uint16_t Onum()    {return m_Onum;}     ///<Returns Orbit number
    uint16_t BID()     {return m_BID;}      ///<Returns Board ID

    uint32_t GEMDAV ()  {return m_GEMDAV;}        ///<Returns GEM DAV list (which chambers have data)
    uint64_t Bstatus()  {return m_Bstatus;}       ///<Returns Buffer status
    int  GDcount()  {return unsigned(m_GDcount);} ///<Returns GEM DAV count (number of chamber blocks)
    uint8_t  Tstate()   {return m_Tstate;}        ///<Returns TTS state

    uint32_t ChamT()    {return m_ChamT;}   ///<Return Chamber Timeout 
    uint8_t  OOSG()     {return m_OOSG;}    ///<Return OOS GLIB (if GLIB is out of sync)

    uint32_t CRC()    {return m_CRC;}
    uint8_t L1AT()    {return m_L1AT;}
    uint32_t DlengthT()    {return m_DlengthT;}

    //!Adds GEB data to vector
    void g_add(GEBdata g){gebd.push_back(g);}
    //!Returns a vector of GEB data
    std::vector<GEBdata> gebs(){return gebd;}
};

