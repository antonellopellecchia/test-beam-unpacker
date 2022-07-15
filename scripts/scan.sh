#!/bin/bash

# Run an efficiency HV scan with multiple runs
# TODO: Support rate scan

RUN_DIR=$1 # $EOS_TESTBEAM
OUT_DIR=$2 # $EOS_WWW/gem/testbeam/analysis
HV_FILE=$3

runs=(00000089 00000090 00000091 00000092 00000094 00000095) # large chamber HV scan
#runs=(00000085 00000084 00000082 00000081 00000080) # tracker HV scan

for run in ${runs[@]}; do
    echo "####################"
    echo "## Analyzing run $run"
    echo "# Unpacker"
    ./RawToDigi $RUN_DIR/raw/$run-{1,0}-0.raw $RUN_DIR/digi/$run.root --geometry may2022Frengo
    #echo "# Local reconstruction"
    #./DigiToRechits $STANDALONE_DIR/digi/$run.root $STANDALONE_DIR/rechits/$run.root
    #echo "# Tracking"
    #./Tracking $STANDALONE_DIR/rechits/$run.root $STANDALONE_DIR/tracks/$run.root
    echo "# Skipping finding noisy channels"
    #python3 analysis/occupancy.py $RUN_DIR/digi/$run.root $OUT_DIR/$run/occupancy --verbose --events 100000 --find-noisy /home/gempro/testbeam/july2022/runs/masks/$run.csv
    echo "# Calculating efficiency and plotting occupancy"
    python3 analysis/occupancy.py $RUN_DIR/digi/$run.root $OUT_DIR/$run/occupancy --verbose --events 100000 --mask-noisy /home/gempro/testbeam/july2022/runs/masks/00000079.csv --efficiency $RUN_DIR/efficiency/${run}_oh.csv $RUN_DIR/efficiency/${run}_chamber.csv
    echo "####################"
    echo ""
    echo ""
done

runs_chamber=( "${runs[@]/%/_chamber.csv}" )
runs_chamber=( "${runs_chamber[@]/#/$RUN_DIR/efficiency/}" )

runs_oh=( "${runs_argument[@]/%/_chamber.csv}" )
runs_oh=( "${runs_argument[@]/#/$RUN_DIR/efficiency/}" )

python3 analysis/scans/efficiency.py $RUN_DIR/efficiency $OUT_DIR/$run/${runs[0]} --hv $HV_FILE
