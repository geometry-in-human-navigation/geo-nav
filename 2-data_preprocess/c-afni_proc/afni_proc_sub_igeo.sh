#!/bin/bash

# Use afni_proc.py to perform motion correction such that 
# the same correction parameter is applied to
# images acquired at different TEs.

# running example: ./afni_proc_sub_igeo.sh sub-01
# tcsh ./preprocessed/proc_sub-01_ME5_e2a

SUBJ=$1

# Note: for ME5, -anat_has_skull no was added after the job has run. So we may want to re-run it. warps anat to epi instead of the other way.
prot="ME5"

project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"
fMRI_data_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_data/spatialfMRI_nii"
freesurfer_dir=${project_dir}/preprocess/freesurfer

mkdir -p ${project_dir}/preprocess/preprocessed/afni_2023

afni_proc.py -subj_id $SUBJ                                                                 \
   -script ${project_dir}/preprocess/preprocessed/proc_${SUBJ}_${prot}_e2a                  \
   -out_dir ${project_dir}/preprocess/preprocessed/afni_2023/${SUBJ}                        \
   -blocks tshift blip align volreg tlrc mask combine surf                                  \
   -copy_anat ${fMRI_data_dir}/${SUBJ}/anat/${SUBJ}_normalized_T1w.nii.gz                   \
   -anat_has_skull yes                                                                      \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-00_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-01_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-02_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-03_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-04_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-05_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-06_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-07_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-08_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-09_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-10_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-11_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-12_acq-${prot}_echo-*_bold.nii.gz           \
   -dsets_me_run                                                                            \
   ${fMRI_data_dir}/$SUBJ/func/${SUBJ}_task-run-13_acq-${prot}_echo-*_bold.nii.gz           \
   -echo_times 14.0 41.51 69.02 96.53 124.04 -reg_echo 1 -align_opts_aea -cost lpc+ZZ       \
   -giant_move -check_flip -skullstrip_opts -blur_fwhm 2                                    \
   -tshift_interp -wsinc9                                                                   \
   -tcat_remove_first_trs 0                                                                 \
   -blip_opts_qw -noXdis -noZdis                                                            \
   -volreg_pvra_base_index MIN_OUTLIER                                                      \
   -volreg_post_vr_allin yes                                                                \
   -volreg_base_ind 7 0                                                                     \
   -volreg_warp_final_interp wsinc5                                                         \
   -align_unifize_epi yes     -align_opts_aea -Allineate_opts -warp shift_rotate            \
   -tlrc_base MNI152_T1_2009c+tlrc -tlrc_NL_warp -mask_epi_anat yes                         \
   -surf_anat ${freesurfer_dir}/subjects/$SUBJ/SUMA/${SUBJ}_SurfVol.nii                     \
   -surf_spec ${freesurfer_dir}/subjects/$SUBJ/SUMA/std.141.${SUBJ}_?h.spec                 \
   -mask_opts_automask -clfrac 0.2 -eclip                                                   \
   -combine_method OC                                                                       \
   -blip_forward_dset ${fMRI_data_dir}/$SUBJ/fmap/${SUBJ}_acq-${prot}_dir-AP_epi.nii.gz     \
   -blip_reverse_dset ${fMRI_data_dir}/$SUBJ/fmap/${SUBJ}_acq-${prot}_dir-PA_epi.nii.gz     \
   -html_review_style pythonic				                                                  
#   -scr_overwrite 
   # -anat_follower_ROI FSvent epi ../freesurfer/ME_benchmark/$SUBJ/SUMA/FT_vent.nii       \
#    -anat_follower_erode FSvent  -volreg_align_e2a



