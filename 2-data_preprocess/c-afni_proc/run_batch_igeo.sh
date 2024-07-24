#!/bin/bash

sub_list="sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-15 sub-16 sub-17 sub-18 sub-19 sub-20"
project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"

for sub in ${sub_list}
do
   ./afni_proc_sub_igeo.sh ${sub}
done

for sub in ${sub_list}
do

   tcsh -xef ${project_dir}/preprocess/preprocessed/proc_${sub}_ME5_e2a 2>&1 | tee ${project_dir}/preprocess/preprocessed/output.proc_${sub}_ME5_e2a

done

