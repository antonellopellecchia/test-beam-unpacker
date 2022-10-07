#!/bin/bash

# Calculate angular corrections
# by iterating track reconstruction
# usage: translation.sh [rechit_file_path] [track_dir] [out_dir] [events] [n_steps] [run_number] [geometry]

RECHIT_FILE=$1
TRACK_DIR=$2
OUT_DIR=$3
EVENTS=$4
N_STEPS=$5
RUN_NUMBER=$6
GEOMETRY=$7

angles=(0. 0. 0. 0.)
iteration=1
while [ "$iteration" -le "$N_STEPS" ]; do
    echo "############################################################"
    iter_track_dir=$TRACK_DIR/$RUN_NUMBER/angles/iteration_$iteration
    mkdir -p $iter_track_dir

    for jchamber in {0..3}; do
        echo "Iteration $iteration, chamber $jchamber"
        track_file=$iter_track_dir/chamber_$jchamber.root
        ./Tracking $RECHIT_FILE $track_file --events $EVENTS --geometry $GEOMETRY

        python3 analysis/residuals.py $track_file $OUT_DIR/iteration_$iteration/chamber_$jchamber

        #corrections_x=($(grep angle $odir/iteration_$iteration/angles.txt | sed "s/angle //g"))
        corrections=($(python3 analysis/utils/get_corrections.py $OUT_DIR/iteration_$iteration/chamber_$jchamber/corrections.txt angle))
         
        echo "Applying correction to chamber $jchamber..."
        echo "Old angles: ${angles[@]}"
        echo "Corrections: ${corrections_x[@]}"
        angles[jchamber]=$( echo "$(printf "%.14f" ${angles[jchamber]}) - $(printf "%.14f" ${corrections[jchamber]})" | bc )
        echo "New angles: ${angles[@]}"

        python3 analysis/utils/correct_geometry.py $GEOMETRY $jchamber angle ${corrections[jchamber]}
        echo "------------------------------------------------------------"
    done

    echo "############################################################"
    echo ""
    echo ""

    iteration=$(( $iteration+1 ))
done

echo "Done iterations."
python3 analysis/utils/plot_alignment.py $OUT_DIR/iteration_*/chamber_* $OUT_DIR/angles
echo "Final corrections: ${angles[@]}"
