#!/bin/bash

# Calculate transversal corrections
# by iterating track reconstruction
# usage: translation.sh [rechit_file_path] [track_dir] [out_dir] [events] [n_steps] [run_number] [geometry]

RECHIT_FILE=$1
TRACK_DIR=$2
OUT_DIR=$3
EVENTS=$4
N_STEPS=$5
RUN_NUMBER=$6
GEOMETRY=$7

translation_x=(0. 0. 0. 0.)
translation_y=(0. 0. 0. 0.)
iteration=1
while [ "$iteration" -le "$N_STEPS" ]; do
    echo "############################################################"
    iter_track_dir=$TRACK_DIR/$RUN_NUMBER/translation/iteration_$iteration
    mkdir -p $iter_track_dir

    for jchamber in {0..3}; do
        echo "Iteration $iteration, chamber $jchamber"
        track_file=$iter_track_dir/chamber_$jchamber.root
        ./Tracking $RECHIT_FILE $track_file --events $EVENTS --geometry $GEOMETRY

        python3 analysis/residuals.py $track_file $OUT_DIR/iteration_$iteration/chamber_$jchamber

        #corrections_x=($(grep angle $odir/iteration_$iteration/angles.txt | sed "s/angle //g"))
        corrections_x=($(python3 analysis/utils/get_corrections.py $OUT_DIR/iteration_$iteration/chamber_$jchamber/corrections.txt x))
        corrections_y=($(python3 analysis/utils/get_corrections.py $OUT_DIR/iteration_$iteration/chamber_$jchamber/corrections.txt y))
         
        echo "Applying correction to chamber $jchamber..."
        echo "Old x: ${translation_x[@]}, old y: ${translation_y[@]}"
        echo "Corrections x: ${corrections_x[@]}"
        echo "Corrections y: ${corrections_y[@]}"
        translation_x[jchamber]=$( echo "$(printf "%.14f" ${translation_x[jchamber]}) + $(printf "%.14f" ${corrections_x[jchamber]})" | bc )
        translation_y[jchamber]=$( echo "$(printf "%.14f" ${translation_y[jchamber]}) + $(printf "%.14f" ${corrections_y[jchamber]})" | bc )
        echo "New x: ${translation_x[@]}, new y: ${translation_y[@]}"

        python3 analysis/utils/correct_geometry.py $GEOMETRY $jchamber x ${corrections_x[jchamber]}
        python3 analysis/utils/correct_geometry.py $GEOMETRY $jchamber y ${corrections_y[jchamber]}
        echo "------------------------------------------------------------"
    done

    echo "############################################################"
    echo ""
    echo ""

    iteration=$(( $iteration+1 ))
done

echo "Done iterations."
python3 analysis/utils/plot_alignment.py $OUT_DIR/iteration_*/chamber_* $OUT_DIR/translation
echo "Final corrections x: ${translation_x[@]}"
echo "Final corrections y: ${translation_y[@]}"
