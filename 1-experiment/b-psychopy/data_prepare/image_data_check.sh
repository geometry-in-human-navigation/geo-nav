#!/bin/bash
#script to loop through directories to check files
cd /media/statespace/Spatial/sptialworkspace/spatialfMRI/simulation/carla_ws/recording/output_synchronized
start_index=10
total_num=8000
end_index=`expr $start_index + $total_num`

for D in ./*; do
    if [ -d "$D" ]; then
        cd "$D"
        img_counter=0
        for f in *.jpg
        do
            # echo -n "$f"
            file_name_suffix=(${f//_/ })
            # echo "${file_name_suffix[1]}"
            name_suffix=(${file_name_suffix[1]//./ })
            current_index=${name_suffix[0]}
            current_index=`expr $current_index - 0`
            # echo "current_index = $current_index" 

            if (( current_index >= start_index )); then
                if (( current_index < end_index ));then
                    img_counter=`expr $img_counter + 1`
                    # echo "current_index = $current_index" 
                fi
            fi
        done

        if (( img_counter >= total_num )); then
            echo "Qualified: $D number of images = $img_counter"
        else
            echo "Nonqualified: $D number of images = $img_counter"
        fi
        cd ..
    fi
done

