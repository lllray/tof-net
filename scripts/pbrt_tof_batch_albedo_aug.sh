#!/bin/bash
#FOLDER=/home/shuochsu/Research/pbrt-v3-tof/Results
#FOLDER_RGB=/home/shuochsu/Research/pbrt-v3
FOLDER=/home/lixin/ROS/DeepEnd2End/DeepToF_release_0.1/src/pbrt-v3-tof/Results
#FOLDER_RGB=/home/shuochsu/Research/pbrt-v3
####### takes the LookAt file from Blender to change camera view #######
DIST="10"
for TYPE in all; do
    for ALBEDO in 10 9 8 7 6 5 4 3 2 1; do
        for SCENE in breakfast contemporary-bathroom pavilion; do #bathroom white-room
            FOLDER_SCENE=/home/lixin/ROS/DeepEnd2End/DeepToF_release_0.1/pbrt-v3-scenes/$SCENE
            FN_CAMERAPATH=$FOLDER_SCENE/"camerapath.txt"
            FNUM=0
            while read -r line; do
                FNUM=$((FNUM+1))
                PROC=$(($FNUM%10))
                if [ "$PROC" -gt 0 ]; then
                    continue
                fi
                OUTPUT=$FOLDER'/'$SCENE'_'$ALBEDO'_'$TYPE'_'$FNUM
                mkdir $OUTPUT
                cd $OUTPUT
                INPUT=/home/lixin/ROS/DeepEnd2End/DeepToF_release_0.1/pbrt-v3-scenes/$SCENE/$SCENE'_'tof_tmp_$FNUM.pbrt
                rm $INPUT
                echo "$line\n$(cat $FOLDER_SCENE/$SCENE'_tof_'$ALBEDO'.pbrt')" > $INPUT
                /home/lixin/ROS/DeepEnd2End/DeepToF_release_0.1/src/pbrt-v3-tof/pbrt_$TYPE'_'$DIST $INPUT
                # $FOLDER_RGB/pbrt $INPUT
                rm $INPUT
            done < "$FN_CAMERAPATH"
        done
    done
done
