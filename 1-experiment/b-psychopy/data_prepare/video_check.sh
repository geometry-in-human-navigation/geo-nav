#!/bin/bash
#script to loop through directories to check files
cd /media/statespace/S/recording/videos

for D in ./*; do
    if [ -d "$D" ]; then
        cd "$D"
        for f in *.mp4
        do
            # echo -n "$f "
            video_length=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -sexagesimal "$f"`
            # echo $video_length
            if [ "$video_length" = "0:03:30.000000" ]; then
                :
                # echo "Video is correct."
            else
                echo -n "$f "
                echo "$video_length"
            fi
        done
        cd ..
    fi
done

