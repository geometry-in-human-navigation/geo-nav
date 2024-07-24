#!/bin/bash

# module load freesurfer

project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"
fMRI_data_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_data/spatialfMRI_nii"
freesurfer_dir=${project_dir}/preprocess/freesurfer

cd ${project_dir}/preprocess/preprocessed/afni_2023
# list=`ls -d sub-*`
# echo $list

sub_list="sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-15 sub-16 sub-17 sub-18 sub-19 sub-20"
echo $sub_list

for sub in ${sub_list}
do
	echo $sub
	cd ${freesurfer_dir}/subjects/$sub
	@SUMA_Make_Spec_FS -sid $sub -NIFTI

	cd ${freesurfer_dir}/subjects/$sub/SUMA
	# 3dcalc -a aseg.auto.nii -expr 'amongst(a,3,8,9,10,11,12,13,16,17,18,26,42,47,48,49,50,51,52,53,54,58)' -prefix neural_mask.nii.gz
	# cp aparc.a2009s+aseg_REN_gm.nii.gz neural_mask.nii.gz
	# cp aparc.a2009s+aseg_REN_all.nii.gz neural_mask.nii.gz
	3dcalc -a aparc.a2009s+aseg_REN_gm.nii.gz -b aparc.a2009s+aseg_REN_wmat.nii.gz -expr 'or(a,b)' -prefix neural_mask.nii.gz -overwrite

	cd ${project_dir}/preprocess/preprocessed/afni_2023/$sub
	@SUMA_AlignToExperiment -exp_anat ${sub}_normalized_T1w+orig.  -surf_anat ${freesurfer_dir}/subjects/$sub/SUMA/${sub}_SurfVol.nii   -align_centers  -overwrite_resp O -followers_interp NN -overwrite
	cat_matvec -ONELINE ${sub}_normalized_T1w_al_keep_mat.aff12.1D ${sub}_SurfVol_Alnd_Exp.A2E.1D > ${sub}_SurfVol_Alnd_Exp.S2E.1D
	3dAllineate -1Dmatrix_apply ${sub}_SurfVol_Alnd_Exp.S2E.1D -input ${freesurfer_dir}/subjects/${sub}/SUMA/${sub}_SurfVol.nii -master anat_final.${sub}+orig. -prefix ${sub}_SurfVol_Alnd_Exp -overwrite
	3dAllineate -1Dmatrix_apply ${sub}_SurfVol_Alnd_Exp.S2E.1D -input ${freesurfer_dir}/subjects/${sub}/SUMA/neural_mask.nii.gz   -master anat_final.${sub}+orig. -prefix neural_mask.nii_Alnd_Exp -overwrite -interp NN
	# @SUMA_AlignToExperiment -exp_anat anat_final.$sub+orig.  -surf_anat ${freesurfer_dir}/subjects/$sub/SUMA/${sub}_SurfVol.nii  -al -align_centers -strip_skull both -overwrite_resp O -followers_interp NN   -surf_anat_followers ${freesurfer_dir}/subjects/${sub}/SUMA/neural_mask.nii.gz  
	
	# 3dfractionize -template pb04.${sub}.r01.combine+orig. -input neural_mask.nii_Alnd_Exp+orig. -clip 0.2 -preserve -prefix neuralmask_resample.nii.gz
	3dfractionize -template pb04.${sub}.r01.combine+orig. -input neural_mask.nii_Alnd_Exp+orig. -clip 0.2 -prefix neuralmask_resample.nii.gz -overwrite
	3dcalc -a full_mask.${sub}+orig. -b neuralmask_resample.nii.gz -expr 'and(a,b)' -prefix final_neural_mask.nii.gz -overwrite

done

cd ${project_dir}
