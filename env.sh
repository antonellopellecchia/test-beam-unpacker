source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh

export ANALYSIS_BUILD=$PWD
export ANALYSIS_HOME=$ANALYSIS_BUILD/../

RUN_DIR=$1
GEOMETRY=$2

function reconstruction() {
    run_number=$1

    zstd --decompress $RUN_DIR/compressed/$run_number-{0,1}-0.raw.zst
    mv $RUN_DIR/compressed/$run_number-{1,0}-0.raw $RUN_DIR/raw/
    ./RawToDigi $RUN_DIR/raw/$run_number-{1,0}-0.raw $RUN_DIR/digi/$run_number.root --geometry $GEOMETRY
    ./DigiToRechits $RUN_DIR/digi/$run_number.root $RUN_DIR/rechits/$run_number.root --geometry $GEOMETRY
    ./Tracking $RUN_DIR/rechits/$run_number.root $RUN_DIR/tracks/$run_number.root --geometry $GEOMETRY
    python3 analysis/ge21.py $RUN_DIR/tracks/$run_number.root $RUN_DIR/results/$run_number/me0_blank --chamber 3
 }
