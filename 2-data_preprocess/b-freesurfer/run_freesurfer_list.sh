#!/usr/bin/env bash

#Convert T1w to surfer
# print date
date

# module load freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# set project directory
project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process/preprocess/freesurfer"

# set freesurfer subjects
mkdir -p ${project_dir}/subjects
subjects_dir=${project_dir}/subjects

# export SUBJECTS_DIR for freesurfer
export SUBJECTS_DIR=$subjects_dir

# This directory should be where the data lives.
fMRI_data_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_data/spatialfMRI_nii"
data_dir=${fMRI_data_dir}
list=(`ls -d ${fMRI_data_dir}/sub-*`)

sub_list="sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-15 sub-16 sub-17 sub-18 sub-19 sub-20"
# sub_list="sub-mni"

for sub in ${sub_list}
do
    subj=${sub}

    # set freesurfer output
    mkdir -p ${project_dir}/output
    out_dir=${project_dir}/output
    mkdir -p $project_dir/logs

    mkdir -p $subjects_dir/$subj/mri/orig/

    # T1file is the T1 image that needs to be converted to mgz file
    3dcopy $data_dir/${subj}/anat/${subj}_normalized_T1w.nii.gz $subjects_dir/$subj/mri/orig/T1w.nii.gz

    mri_convert $subjects_dir/$subj/mri/orig/T1w.nii.gz $subjects_dir/$subj/mri/orig/001.mgz

    # rm $subjects_dir/$subj/scripts/IsRunning.lh+rh
    time recon-all -all -subjid $subj -openmp 32 > $project_dir/logs/fs_$subj.log

    @SUMA_Make_Spec_FS -NIFTI -fspath $subjects_dir/$subj/ -sid $subj > $project_dir/logs/suma_$subj.log

    mkdir -p $out_dir/$subj/T1w/
    mri_convert $subjects_dir/$subj/mri/orig/001.mgz $out_dir/$subj/T1w/brainmask.nii.gz

    @auto_tlrc -no_ss -input $out_dir/$subj/T1w/brainmask.nii.gz -base MNI152_T1_2009c+tlrc > $project_dir/logs/tlrc_$subj.log

    echo "FINISHED"
    date

done