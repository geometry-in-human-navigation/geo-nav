#!/bin/bash

source /home/statespace/anaconda3/etc/profile.d/conda.sh
conda activate mybrainiak

cd /media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process/process/shared_glm

start=$(date +%s)

# sleep 2
python iterate_vit_shared_dims.py > iterate_vit_shared_output20230803.txt

end=$(date +%s)

echo "Elapsed Time: $(($end-$start)) seconds"
secs=$end-$start
echo "Elapsed Time: $((secs/3600))h:$((secs%3600/60))m:$((secs%60))s"
