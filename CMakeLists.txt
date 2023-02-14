
#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(unpacker)

find_package(ROOT REQUIRED Minuit2)
include(${ROOT_USE_FILE})

include_directories($ENV{ANALYSIS_HOME}/include)

set(sources
    $ENV{ANALYSIS_HOME}/src/StripMapping.cc
    $ENV{ANALYSIS_HOME}/src/ChamberMapping.cc
    $ENV{ANALYSIS_HOME}/src/SetupGeometry.cc
    $ENV{ANALYSIS_HOME}/src/DataFrame.cc
    $ENV{ANALYSIS_HOME}/src/Track2D.cc
    $ENV{ANALYSIS_HOME}/src/Track.cc
    $ENV{ANALYSIS_HOME}/src/LinearFcn.cc
    $ENV{ANALYSIS_HOME}/src/DetectorTracker.cc
    $ENV{ANALYSIS_HOME}/src/DetectorLarge.cc
    $ENV{ANALYSIS_HOME}/src/Hit.cc
    $ENV{ANALYSIS_HOME}/src/Rechit2D.cc
    $ENV{ANALYSIS_HOME}/src/Rechit.cc
    $ENV{ANALYSIS_HOME}/src/Cluster.cc
    $ENV{ANALYSIS_HOME}/src/Digi.cc)

set(header $ENV{ANALYSIS_HOME}/include/*.h)

#----------------------------------------------------------------------------
# Locate sources and headers for the unpacker
# NB: headers are included so they will show up in IDEs
#
# file(GLOB headers $ENV{ANALYSIS_HOME}/include/Cluster.h)
# file(GLOB headers $ENV{ANALYSIS_HOME}/include/Digi.h)

# file(GLOB headers $ENV{ANALYSIS_HOME}/include/Event.h)
# file(GLOB headers $ENV{ANALYSIS_HOME}/include/GEMAMCEventFormat.h)
# file(GLOB sources $ENV{ANALYSIS_HOME}/src/TestBeamMapping.cc)
# file(GLOB sources $ENV{ANALYSIS_HOME}/src/Cluster.cc)
# file(GLOB sources $ENV{ANALYSIS_HOME}/src/Digi.cc)

#----------------------------------------------------------------------------
# Add the executable
#
add_executable(RawToDigi unpacker.cc ${sources} ${headers})
target_link_libraries(RawToDigi ${ROOT_LIBRARIES})

add_executable(DigiToRechits DigiToRechits.cc ${sources} ${headers})
target_link_libraries(DigiToRechits ${ROOT_LIBRARIES})

add_executable(Tracking Tracking.cc ${sources} ${headers})
target_link_libraries(Tracking ${ROOT_LIBRARIES})

add_executable(Alignment Alignment.cc ${sources} ${headers})
target_link_libraries(Alignment ${ROOT_LIBRARIES})

#----------------------------------------------------------------------------
# Copy script and analysis files
#
configure_file(mapping/oct2021/tracker_mapping.csv mapping/oct2021/tracker_mapping.csv COPYONLY)
configure_file(mapping/oct2021/ge21_mapping.csv mapping/oct2021/ge21_mapping.csv COPYONLY)
configure_file(mapping/oct2021/chamber_mapping.csv mapping/oct2021/chamber_mapping.csv COPYONLY)
configure_file(mapping/nov2021/me0_mapping.csv mapping/oct2021/me0_mapping.csv COPYONLY)
configure_file(mapping/nov2021/20x10_mapping.csv mapping/oct2021/20x10_mapping.csv COPYONLY)

configure_file(mapping/nov2021/tracker_mapping.csv mapping/nov2021/tracker_mapping.csv COPYONLY)
configure_file(mapping/nov2021/ge21_mapping.csv mapping/nov2021/ge21_mapping.csv COPYONLY)
configure_file(mapping/nov2021/me0_mapping.csv mapping/nov2021/me0_mapping.csv COPYONLY)
configure_file(mapping/nov2021/20x10_mapping.csv mapping/nov2021/20x10_mapping.csv COPYONLY)
configure_file(mapping/nov2021/chamber_mapping.csv mapping/nov2021/chamber_mapping.csv COPYONLY)

configure_file(mapping/may2022/tracker_mapping.csv mapping/may2022/tracker_mapping.csv COPYONLY)
configure_file(mapping/may2022/ge21_mapping.csv mapping/may2022/ge21_mapping.csv COPYONLY)
configure_file(mapping/may2022/me0_mapping.csv mapping/may2022/me0_mapping.csv COPYONLY)
configure_file(mapping/may2022/chamber_mapping.csv mapping/may2022/chamber_mapping.csv COPYONLY)

configure_file(mapping/july2022/tracker_mapping.csv mapping/july2022/tracker_mapping.csv COPYONLY)
configure_file(mapping/july2022/me0_mapping.csv mapping/july2022/me0_mapping.csv COPYONLY)
configure_file(mapping/july2022/chamber_mapping.csv mapping/july2022/chamber_mapping.csv COPYONLY)
configure_file(mapping/july2022/template.csv mapping/july2022/template.csv COPYONLY)

configure_file(mapping/me0stack/me0_mapping.csv mapping/me0stack/me0_mapping.csv COPYONLY)
configure_file(mapping/me0stack/chamber_mapping.csv mapping/me0stack/chamber_mapping.csv COPYONLY)

configure_file(mapping/generate.py mapping/generate.py COPYONLY)
configure_file(mapping/vfat.csv mapping/vfat.csv COPYONLY)

configure_file(geometry/may2022.csv geometry/may2022.csv COPYONLY)
configure_file(geometry/july2022.csv geometry/july2022.csv COPYONLY)
configure_file(geometry/me0stack.csv geometry/me0stack.csv COPYONLY)

configure_file(analysis/residuals.py analysis/residuals.py COPYONLY)
configure_file(analysis/occupancy.py analysis/occupancy.py COPYONLY)
configure_file(analysis/profile.py analysis/profile.py COPYONLY)
configure_file(analysis/efficiency.py analysis/efficiency.py COPYONLY)
configure_file(analysis/ge21.py analysis/ge21.py COPYONLY)
configure_file(analysis/strips.py analysis/strips.py COPYONLY)
configure_file(analysis/horns.py analysis/horns.py COPYONLY)
configure_file(analysis/scan.py analysis/scan.py COPYONLY)
configure_file(analysis/scans/rate.py analysis/scans/rate.py COPYONLY)
configure_file(analysis/scans/efficiency.py analysis/scans/efficiency.py COPYONLY)
configure_file(analysis/track_angles.py analysis/track_angles.py COPYONLY)

configure_file(analysis/scans/efficiency.py analysis/scans/efficiency.py COPYONLY)
configure_file(analysis/calibration/latency.py analysis/calibration/latency.py COPYONLY)
configure_file(analysis/calibration/threshold.py analysis/calibration/threshold.py COPYONLY)
configure_file(analysis/calibration/rate.py analysis/calibration/rate.py COPYONLY)

configure_file(scripts/alignment/angles.sh scripts/alignment/angles.sh COPYONLY)
configure_file(scripts/alignment/translation.sh scripts/alignment/translation.sh COPYONLY)
configure_file(scripts/scan.sh scripts/scan.sh COPYONLY)

configure_file(masks/may2022.csv masks/may2022.csv COPYONLY)

configure_file(analysis/utils/get_corrections.py analysis/utils/get_corrections.py COPYONLY)
configure_file(analysis/utils/plot_alignment.py analysis/utils/plot_alignment.py COPYONLY)
configure_file(analysis/utils/correct_geometry.py analysis/utils/correct_geometry.py COPYONLY)

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS RawToDigi DESTINATION bin)
install(TARGETS DigiToRechits DESTINATION bin)
install(TARGETS Tracking DESTINATION bin)
