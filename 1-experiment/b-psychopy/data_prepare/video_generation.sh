#!/bin/bash

recording_dir="/media/statespace/S/recording/output_synchronized/*"
output_video_dir="/home/statespace/Workspace/simulation/carla_ws/recording/spatialcross_video"
# mkdir -p ${output_video_dir}
town="Town10"

# clip_size=300 # 300 seconds, 5 min
# clip_fps=50
# clip_frames=${clip_size}*${clip_fps}
clip_frames=15000

for f in $recording_dir; do
    if [ -d "$f" ]; then
        if [[ "$f" == *"$town"* ]]; then

            cd $f
            echo "cd $f"

            town_name=${f##*/}
            echo $town_name

            clip_index=0
            clip_start_number=`expr ${clip_index} \* ${clip_frames} + 60`
            ffmpeg -y -framerate 50 -start_number ${clip_start_number} -i rgb_%06d.jpg  -vframes ${clip_frames}  ${output_video_dir}/${town_name}.mp4 
        fi
    fi
done