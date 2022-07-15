#include <cstdio>
#include <iostream>
#include <cstdint>
#include <vector>
#include <array>
#include <utility>
#include <bitset>

#include <zstd.h>

union EventHeader {
    struct {
        uint64_t word1;
        uint64_t word2;
        uint64_t word3;
    };
    struct {
        uint16_t version;
        uint16_t flags;
        uint32_t run_number;
        uint32_t lumisection;
        uint32_t event_number;
        uint32_t event_size;
        uint32_t crc32c;
    };
};

void printBinary(void *raw, uint8_t nwords) {
    for (int iword=0; iword<nwords; iword++) {
        uint64_t word = ((uint64_t *) raw)[iword];
        std::cout << "\t\t" << std::bitset<64>(word) << std::endl;
    }
}

void printHeader(union EventHeader header, bool raw) {
    if (raw) {
        std::cout << "\t\theader word 1 " << std::bitset<64>(header.word1) << std::endl;
        std::cout << "\t\theader word 2 " << std::bitset<64>(header.word2) << std::endl;
        std::cout << "\t\theader word 3 " << std::bitset<64>(header.word3) << std::endl;
    } else {
        std::cout << "\t\t";
        std::cout << "version " << header.version << ", ";
        std::cout << "flags " << header.flags << ", ";
        std::cout << "run_number " << header.run_number << ", ";
        std::cout << "lumisection " << header.lumisection << ", ";
        std::cout << "event_number " << header.event_number << ", ";
        std::cout << "event_size " << header.event_size << ", ";
        std::cout << "crc32c " << header.crc32c;
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{

    std::cout << "Running zstd debugger..." << std::endl;
    if (argc < 2) {
	std::cout <<
	    "Usage: decompression compressed_file decompressed_file [--words nwords]"
	    << std::endl;
	return 0;
    }
    std::vector < std::string > ifiles;

    int nwords = -1;
    bool isUnnamed = true;
    for (int iarg = 1; iarg < argc; iarg++) {
        std::string arg = argv[iarg];
        if (arg[0] == '-') {	// parse named parameters
            isUnnamed = false;	// end of unnamed parameters
            if (arg == "--words")
            nwords = atoi(argv[iarg + 1]);
        } else if (isUnnamed) {	// unnamed parameters
            ifiles.push_back(arg);
        }
    }
    std::cout << "ifiles ";
    for (auto s:ifiles) std::cout << s << " ";
    std::cout << std::endl;

    std::vector < std::FILE * >m_files;

    std::cout << "Reading input files..." << std::endl;
    try {
      for (auto ifilename:ifiles) m_files.push_back(std::fopen(ifilename.c_str(), "rb"));
    }
    catch(int e) {
        std::cout << "An exception occured. Exception code " << e << std::endl;
    }
    std::cout << "Input files opened." << std::endl;


    if (nwords > 0)
        std::cout << "Analyzing " << nwords << " words" << std::endl;
    else
        std::cout << "Analyzing all words" << std::endl;


    int read_words = 0;


    ZSTD_DStream *decompressionStream = ZSTD_createDStream();
    //std::size_t buffInSize = ZSTD_initDStream(decompressionStream);

    size_t buffInSize = ZSTD_DStreamInSize();
    size_t buffOutSize = ZSTD_DStreamOutSize();
    std::cout << "Recommended input size: " << buffInSize << std::endl;
    std::cout << "Recommended output size: " << buffOutSize << std::endl;
    
    void *buffIn = malloc(buffInSize);
    void *buffOut = malloc(buffOutSize);

    ZSTD_inBuffer input = { buffIn, buffInSize, 0 };
    ZSTD_outBuffer output = { buffOut, buffOutSize, 0 };

    bool isError;
    size_t remaining;
    int iwords = 0;
    bool isOutputBufferEnd = true;
    bool isInputBufferEnd = true;
    
    void *word = malloc(sizeof(uint64_t)); // buffer 8-byte word to contain decompressed data 
    int buffOutIndex = 0; // where to get the data word from the output buffer
    int bitsInBuffer = 0; // where to get the data word from the output buffer

    while (true) {
        
        if ((nwords > 0) && (read_words > nwords)) break;

        std::cout << "Word " << read_words << std::endl;

        union EventHeader hdr;

        for (int ifile = 0; ifile < m_files.size(); ifile++) {

            if (ifile % 2 == 0) {
            // read compressed file:

                std::cout << "\tCompressed file:" << std::endl;

                do {
                   
                    if (isInputBufferEnd) {
                        std::cout << "\t\tInput buffer flushed, reading again from file..." << std::endl;
                        std::size_t sz = std::fread(buffIn, buffInSize, 1, m_files.at(ifile));
                        isInputBufferEnd = false;
                    }
                    if (isOutputBufferEnd) {
                        std::cout << "\t\tOutput buffer flushed, decompressing remaining input buffer..." << std::endl;
                        std::cout << "\t\tNew input size " << input.size << std::endl;
                       
                        // flush output to get remaining data there:
                        remaining = ZSTD_decompressStream(decompressionStream, &output, &input);
                        isError = (bool) ZSTD_isError(remaining);
                        //input.size = remaining;
                        //buffIn = malloc(remaining);
                   
                        buffOutIndex = 0;
                    }

                    // print buffer check variables:
                    std::cout << "\t\t";
                    std::cout << "input: size " << input.size << ", position " << input.pos << ", " ;
                    std::cout << "output: size " << output.size << ", position " << output.pos << ", ";
                    std::cout << "remaining " << remaining;
                    if (isError) std::cout << ". Error: " << ZSTD_getErrorName(remaining);
                    std::cout << std::endl;
                    
                    bitsInBuffer = buffOutIndex*8;
                    std::cout << "\t\tbits read from output buffer: " << bitsInBuffer << std::endl;

                    // just get the next word from the output buffer:
                    word = (void *) ((uint64_t *) buffOut)[buffOutIndex];
                    buffOutIndex++;

                    // update size of next compressed input:
                    //input.size += remaining;

                    //finished = (input.pos == input.size)
                    iwords++;
                    isOutputBufferEnd = bitsInBuffer+8>=output.size;
                
                } while (iwords<1 && !isError);// && remaining>0 && !isError && input.pos>=input.size);

                std::cout << "\t\t" << std::bitset<64>((uint64_t) word) << std::endl;
                //printBinary(word, 1);
                //printHeader(hdr, true);
                //printHeader(hdr, false);

            } else {
                // read decompressed file:
                
                std::cout << "\tDecompressed file:" << std::endl;
                std::size_t sz = std::fread(&word, sizeof(uint64_t), 1, m_files.at(ifile));
                if (sz == 0 ) return -1; // end of file reached

                std::cout << "\t\t" << std::bitset<64>((uint64_t) word) << std::endl;
                //printHeader(hdr, true);
                //printHeader(hdr, false);
            }
        }
        read_words++;
    }

    std::cout << std::endl;

}
