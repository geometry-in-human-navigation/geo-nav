### mask epi data
#### set mask epi data env
import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 
import copy

import numpy as np
from scipy import stats
import scipy.spatial.distance as sp_distance
from sklearn.svm import NuSVC

from brainiak.isc import isc
from brainiak.fcma.util import compute_correlation
import brainiak.funcalign.srm
from brainiak import image, io
import brainiak

import nibabel as nib
import nilearn as nil
import glob
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.image import concat_imgs, index_img

import matplotlib.pyplot as plt
from einops import rearrange
import datetime

# debug
glm_show_alpha = False

project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"
fMRI_data_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_data/spatialfMRI_nii"

preprocessed_dir=project_dir+"/preprocess/preprocessed/afni_2023"
processed_dir=project_dir+"/process/processed"

num_runs = 14
num_TRs = 223
num_midlayer_units = 8
num_layer_pca_components = 20
num_deeplayers_units = num_midlayer_units + 12*num_layer_pca_components
num_semantic_categories = 10

start_fixation_TRs = 3
hemodynamic_shift_TRs = 3
num_TRs_video = 200

#### load atlas brain regions
import os
# read design matrix
atlas_labels_txt = "Atlas MNI_Glasser_HCP_v1.0, 360 regions.txt"
atlas_labels_file = os.path.join(project_dir, "process/shared_glm", atlas_labels_txt)

in_file = open(atlas_labels_file,'r')
brain_region_dict={}

# distinguish left and right brain regions
atlas_all_lines = in_file.readlines()

for i_line in range(0,len(atlas_all_lines)):
    if "u:L_" in atlas_all_lines[i_line]:
        
        atlas_brain_region_label = []

        # left brain regions
        atlas_line_left = atlas_all_lines[i_line]
        atlas_temp = atlas_line_left.split(":")
        atlas_brain_region_name = atlas_temp[1][2:]
        # print(atlas_brain_region_name)
        atlas_brain_region_label_left = int(atlas_temp[2][:-1])
        atlas_brain_region_label.append(atlas_brain_region_label_left)
        
        # right brain regions
        atlas_line_right = atlas_all_lines[i_line+1]
        atlas_temp = atlas_line_right.split(":")
        atlas_brain_region_name = atlas_temp[1][2:]
        atlas_brain_region_label_right = int(atlas_temp[2][:-1])
        atlas_brain_region_label.append(atlas_brain_region_label_right)

        # put into dict
        brain_region_dict[atlas_brain_region_name] = atlas_brain_region_label

# print(len(brain_region_dict),brain_region_dict)

brain_region_name_list = list(brain_region_dict.keys())
brain_region_label_list = list(brain_region_dict.values())

folder_list = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
               "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", 
               "sub-11", "sub-12", "sub-13", "sub-14", "sub-15",
               "sub-16", "sub-17", "sub-18", "sub-19", "sub-20",
               ]

num_subs = len(folder_list)

##### load
import hdf5storage

file_dir = "/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process/process/shared_glm/temp_data/"
matdic = hdf5storage.loadmat(file_dir+"sub_masked_epi_data_list.mat", )

sub_masked_epi_data_list = matdic["sub_masked_epi_data_list"]

print("number of subjects:", len(sub_masked_epi_data_list))
print("num of brain regions:", len(sub_masked_epi_data_list[0]))

# for i in range(0, len(brain_region_name_list)):
#     print(folder_list[0],',',brain_region_name_list[i],':',sub_masked_epi_data_list[0][i].shape)

### set feature dimensions

#### set feature dimensions by The Optimal Hard Threshold
# The optimal hard threshold for singular values is 4 / sqrt(3)
# 1.https://github.com/brainiak/brainiak/blob/ee093597c6c11597b0a59e95b48d2118e40394a5/brainiak/reprsimil/brsa.py#L157
# 2.http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6846297

from brainiak.reprsimil.brsa import Ncomp_SVHT_MG_DLD_approx

regions_average_dims_list  = []

for i_region in range(0, len(brain_region_name_list)):

  region_dims_list = []

  for i_sub in range(0, num_subs):

    temp_masked_epi_data = sub_masked_epi_data_list[i_sub][i_region]
    ncomp = Ncomp_SVHT_MG_DLD_approx(temp_masked_epi_data.T) # 2-D numpy array of size [n_T, n_V]
    region_dims_list.append(ncomp)

  average_dims = np.mean(np.asarray(region_dims_list))
  regions_average_dims_list.append(np.floor(average_dims).astype(int))

regions_average_dims_array = np.asarray(regions_average_dims_list)
print("regions_average_dims_array:",regions_average_dims_array)
print("features in total:", np.sum(regions_average_dims_array))
# plt.hist(regions_average_dims_array, bins=20)
# plt.show()

###-save all dims mat for future analysis
# equal division of total dims
num_total_dims = np.sum(regions_average_dims_array) # average each subject regions from Ncomp_SVHT_MG_DLD_approx
num_total_regions = 180 # 180 together
num_samples = 100

brain_region_dim_equal_div_array = np.linspace(num_total_regions, num_total_dims, num_samples)

brain_region_dim_scale_list = []
for i in brain_region_dim_equal_div_array:
  brain_region_dim_scale_list.append(num_total_dims/i)

# brain_region_dim_scale_list = brain_region_dim_scale_list[1:] # remove 180 
num_total_dims_list = []

# scale dims
for i_brain_region_dim_scale in brain_region_dim_scale_list:

  brain_region_dim_scale = i_brain_region_dim_scale 
  print(datetime.datetime.now(), ", brain_region_dim_scale:", brain_region_dim_scale)
  region_voxels_array = np.rint(copy.deepcopy(regions_average_dims_array)/brain_region_dim_scale).astype(int)
  num_total_dims = np.sum(region_voxels_array)
  # print("dims in total (with zero dims):", num_total_dims)
  # print("dims in total:", num_total_dims, ",", region_voxels_array)

  region_voxels_array[region_voxels_array == 0] = 1
  srm_dim_max = np.max(region_voxels_array)
  srm_dim_min = np.min(region_voxels_array)

  num_total_dims = np.sum(region_voxels_array)
  print("dims in total:", num_total_dims, ",", region_voxels_array)
  print("max:", srm_dim_max, ",min", srm_dim_min)

  num_total_dims_list.append(num_total_dims)

num_total_dims_array = np.asarray(num_total_dims_list)

print("num_total_dims_array:", num_total_dims_array)

from scipy.io import savemat

matdic = {"num_total_dims_array":num_total_dims_array}
file_dir = project_dir+"/process/shared_glm/temp_data/"
savemat(file_dir + "vit_semgeo_shared_glm_dims_list_optimal_hard_threshold.mat", matdic)
#------------------------------------------------

# scale dims
for i_brain_region_dim_scale in brain_region_dim_scale_list:

  brain_region_dim_scale = i_brain_region_dim_scale 
  print(datetime.datetime.now(), ", brain_region_dim_scale:", brain_region_dim_scale)
  region_voxels_array = np.rint(copy.deepcopy(regions_average_dims_array)/brain_region_dim_scale).astype(int)
  num_total_dims = np.sum(region_voxels_array)
  # print("dims in total (with zero dims):", num_total_dims)
  # print("dims in total:", num_total_dims, ",", region_voxels_array)

  region_voxels_array[region_voxels_array == 0] = 1
  srm_dim_max = np.max(region_voxels_array)
  srm_dim_min = np.min(region_voxels_array)

  num_total_dims = np.sum(region_voxels_array)
  print("dims in total:", num_total_dims, ",", region_voxels_array)
  print("max:", srm_dim_max, ",min", srm_dim_min)

  ### training all data
  train_data = []

  for i_region in range(0, len(brain_region_name_list)):

    each_region_train_data = []

    for i_sub in range(0, num_subs):

      temp_masked_epi_data = sub_masked_epi_data_list[i_sub][i_region]

      each_region_train_data.append(temp_masked_epi_data)

    train_data.append(each_region_train_data)

  srm_list = []
  for i_region in range(0, len(brain_region_name_list)):

    temp_train_data = train_data[i_region]
    n_iter = 200  # iterations of fitting

    features = region_voxels_array[i_region]
    # Create the SRM object
    srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)

    # Fit the SRM data
    # print(brain_region_name_list[i_region], ' Fitting...')
    srm.fit(temp_train_data)
    # print(brain_region_name_list[i_region], ' SRM has been fit.')

    srm_list.append(srm)
    
  ### save shared response
  from scipy.io import savemat
  import copy

  semgeo_shared_region_glm_list = []

  for i_region in range(0, len(brain_region_name_list)):
    shared_response = copy.deepcopy(srm_list[i_region].s_)
    # print(brain_region_name_list[i_region], shared_response.shape)
    semgeo_shared_region_glm_list.append(shared_response)

  matdic = {"semgeo_shared_region_glm_list":semgeo_shared_region_glm_list}
  file_dir = project_dir+"/process/shared_glm/temp_data/"
  savemat(file_dir + "vit_"+ str(num_total_dims) + "_dims" +"_semgeo_shared_region_glm_list_optimal_hard_threshold.mat", matdic)

  # the 3rd part

  ### data preparation
  ##### load design matrix
  from scipy.io import loadmat

  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"
  processed_dir=project_dir+"/process/processed/autoencoder"
  file_dir = os.path.join(processed_dir, "design_matrix")

  matdic = loadmat(os.path.join(file_dir, "design_matrix_alltowns_vit_pca_deeplayers_12layer_20_1latent_8.mat"))

  design_matrix_alltowns = matdic["design_matrix_alltowns"]
  print("design matrix for all towns (n_TRs, n_latent_units) =", design_matrix_alltowns.shape)

  if(np.isnan(design_matrix_alltowns).any()):
      print("design_matrix_alltowns contain NaN values")
      
  ## may not add zscore
  design_matrix_alltowns = stats.zscore(design_matrix_alltowns, axis=0)

  ##### load shared response
  # shared response with SRM feature dimensions determined by number of voxels in each region
  from scipy.io import loadmat

  file_dir = project_dir+"/process/shared_glm/temp_data/"
  matdic = loadmat(file_dir+"vit_"+ str(num_total_dims) + "_dims" +"_semgeo_shared_region_glm_list_optimal_hard_threshold.mat")

  semgeo_shared_region_glm_list = matdic["semgeo_shared_region_glm_list"]
  semgeo_shared_region_glm_list = semgeo_shared_region_glm_list[0]
  print("shared response (shared_features, n_TRs) =", semgeo_shared_region_glm_list[0].shape)

  # zscore shared response - non-effective
  from scipy import stats

  for i_brain_region in range(len(semgeo_shared_region_glm_list)):
    semgeo_shared_region_glm_list[i_brain_region] = stats.zscore(semgeo_shared_region_glm_list[i_brain_region], axis=1)

    if(np.isnan(semgeo_shared_region_glm_list[i_brain_region]).any()):
        print("semgeo_shared_region_glm_list[i_brain_region] contain NaN values")

  ## the encoding model without repeated Town07

  ### train the semantic model
  #### split training and testing data
  # take only one run for one town
  num_run = 14
  num_towns = 8

  Y_train_list = []
  Y_test_list = []

  # take one run as test
  for i_test_run in np.arange(num_towns):

    # spliting into train and test data
    Y_train_regions_list = []
    Y_test_regions_list = []
    Y_train_regions_array = np.empty((1400,0), float)
    Y_test_regions_array = np.empty((200,0), float)

    for i_region in range(0, len(semgeo_shared_region_glm_list)):

      temp_shared_response = semgeo_shared_region_glm_list[i_region]

      Y_data_temp = rearrange(temp_shared_response, 'i (j k) -> i j k', j=num_run)

      Y_data_array = np.concatenate((Y_data_temp[:,0:7,start_fixation_TRs+hemodynamic_shift_TRs:start_fixation_TRs+hemodynamic_shift_TRs+num_TRs_video], 
                Y_data_temp[:,num_run-1:num_run,start_fixation_TRs+hemodynamic_shift_TRs:start_fixation_TRs+hemodynamic_shift_TRs+num_TRs_video]), axis=1)

      Y_train = Y_data_array[:,np.arange(num_towns)!=i_test_run,:]
      Y_test = Y_data_array[:,np.arange(num_towns)==i_test_run,:]

      Y_train = rearrange(Y_train, 'i j k -> i (j k) ')
      Y_test = rearrange(Y_test, 'i j k -> i (j k) ')

      Y_train = Y_train.T
      Y_test = Y_test.T

      Y_train_regions_array = np.append(Y_train_regions_array, Y_train, axis=1)
      Y_test_regions_array = np.append(Y_test_regions_array, Y_test, axis=1)

    Y_train_list.append(Y_train_regions_array)
    Y_test_list.append(Y_test_regions_array)

  print("Y_train (n_TRs, shared_features) =", Y_train.shape)
  print("Y_test (n_TRs, shared_features) =", Y_test.shape)

  print("Y_train_regions_array (n_TRs, n_regions*shared_features) =", Y_train_regions_array.shape)
  print("Y_test_regions_array (n_TRs, n_regions*shared_features) =", Y_test_regions_array.shape)

  if(np.isnan(Y_train_regions_array).any()):
      print("Y_test_regions_array contain NaN values")

  if(np.isnan(Y_test_regions_array).any()):
      print("Y_test_regions_array contain NaN values")

  X_data_temp = []
  for i_run in np.arange(num_towns):
      data_temp = design_matrix_alltowns[223*i_run:223*(i_run+1),-(num_deeplayers_units+num_semantic_categories):] # first 8, last num_semantic_categories
      data_temp = data_temp[start_fixation_TRs+hemodynamic_shift_TRs:start_fixation_TRs+hemodynamic_shift_TRs+num_TRs_video,
                                          -num_semantic_categories:]
      X_data_temp.append(data_temp)

  X_data_temp = np.asarray(X_data_temp)

  X_train_list = []
  X_test_list = []

  for i_test_run in np.arange(num_towns):

      X_train = X_data_temp[np.arange(num_towns)!=i_test_run,:,:]
      X_test = X_data_temp[np.arange(num_towns)==i_test_run,:,:]

      X_train = rearrange(X_train, 'i j k -> (i j) k')
      X_test = rearrange(X_test, 'i j k -> (i j) k')

      X_train_list.append(X_train)
      X_test_list.append(X_test)

      if(np.isnan(X_train).any()):
          print("X_train contain NaN values")

      if(np.isnan(X_test).any()):
          print("X_test contain NaN values")

  print("X_train (n_TRs, n_latent_units) =", X_train.shape)
  print("X_test (n_TRs, n_latent_units) =", X_test.shape)

  #### fit the model
  score_subs_list = []
  Y_test_orig_list = []
  Y_test_pred_list = []

  for i_test_run in np.arange(num_towns):

      print("i_run:",i_test_run)
      
      # get X data
      X_train = X_train_list[i_test_run]
      X_test = X_test_list[i_test_run]

      # get Y data
      Y_train = Y_train_list[i_test_run]
      Y_test = Y_test_list[i_test_run]

      # import
      from sklearn.model_selection import check_cv
      from voxelwise_tutorials.utils import generate_leave_one_run_out

      # indice of first sample of each run
      run_onsets = []
      num_run_train=7
      for i in range(num_run_train):
          run_onsets.append(i*num_TRs_video)

      n_samples_train = X_train.shape[0]
      cv = generate_leave_one_run_out(n_samples_train, run_onsets)
      cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler(with_mean=False, with_std=False)

      from voxelwise_tutorials.delayer import Delayer
      delayer = Delayer(delays=[0])

      from himalaya.kernel_ridge import KernelRidgeCV
      from himalaya.ridge import RidgeCV
      from himalaya.backend import set_backend
      
      backend = set_backend("torch_cuda", on_error="warn")

      alphas = np.logspace(-3, 20, 120)
      kernel_ridge_cv = RidgeCV(
          alphas=alphas, cv=cv, fit_intercept=True,
          solver_params=None) # try None
          # solver_params=dict(n_targets_batch=500, n_alphas_batch=5, 
          #                 n_targets_batch_refit=100)) # try None

      from sklearn.pipeline import make_pipeline
      pipeline = make_pipeline(
          scaler,
          delayer,
          kernel_ridge_cv,
      )
      from sklearn import set_config
      set_config(display='diagram')  # requires scikit-learn 0.23
      _ = pipeline.fit(X_train, Y_train)

      # test
      scores = pipeline.score(X_test, Y_test)
      scores = backend.to_numpy(scores)
      score_subs_list.append(scores)

      # predict
      Y_test_predicted = pipeline.predict(X_test)
      Y_test_orig_list.append(Y_test)
      Y_test_pred_list.append(Y_test_predicted)

      # plot best alphas
      if glm_show_alpha == True:
          from himalaya.viz import plot_alphas_diagnostic
          import matplotlib.pyplot as plt
          best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
          print('best_alphas:',best_alphas.shape)
          plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
          plt.show()

  score_subs_array = np.array(score_subs_list)
  print("score_subs_array.shape:",score_subs_array.shape)

  Y_test_orig_semantic = np.asarray(Y_test_orig_list)
  print("Y_test_orig_semantic.shape:",Y_test_orig_semantic.shape)

  Y_test_pred_semantic = np.asarray(Y_test_pred_list)
  print("Y_test_pred_semantic.shape:",Y_test_pred_semantic.shape)

  #### save ev to nifti for visulization 
  ##### ev on cross all runs
  from sklearn.metrics import r2_score

  # ve for each brain regions
  dim_accumulation = 0
  sub_score = np.zeros((len(brain_region_name_list),), dtype=float)

  for i_region in range(0, len(brain_region_name_list)):
      dim_features = region_voxels_array[i_region]

      error = 0
      var = 0
      for i_test_run in np.arange(num_towns):

          Y_test = Y_test_orig_list[i_test_run]
          Y_test_predicted = Y_test_pred_list[i_test_run]

          y_true = Y_test[:, dim_accumulation:dim_accumulation + dim_features]
          y_pred = Y_test_predicted[:, dim_accumulation:dim_accumulation + dim_features]
          
          error = error + ((y_true - y_pred) ** 2.0).sum()
          var = var + ((y_true - y_true.mean(0)) ** 2.0).sum()

      dim_accumulation = dim_accumulation + dim_features

      sub_score[i_region] = 1.0 - error / var
      
  # show summary ev runs
  explained_variance_semantic = copy.deepcopy(np.asarray(sub_score))
  print('score shape:',explained_variance_semantic.shape)

  ##### save and load mat
  from scipy.io import savemat
  import copy

  print("explained_variance_semantic shape:", explained_variance_semantic.shape)

  matdic = {"sub_score": explained_variance_semantic}
  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"

  file_dir = project_dir+"/process/shared_glm/temp_data/"
  savemat(file_dir+ "vit_"+ str(num_total_dims) + "_dims" + "_semantic_ev_score_brain_regions.mat", matdic)

  #### permutation before cross-validation

  ##### the semantic permutation model
  # ve for each brain regions
  shuffle_num_times = 2
  sub_score_shuffle_list = []

  sub_score = np.zeros((len(brain_region_name_list), shuffle_num_times,), dtype=float)

  for i_shuffle in range(shuffle_num_times):

      np.random.seed(i_shuffle)
      single_run_TRs_video_indices = np.arange(num_TRs_video)
      X_test_indices_2 = np.arange(X_test_list[0].shape[1])
      Y_test_indices_2 = np.arange(Y_test_list[0].shape[1])
      np.random.shuffle(single_run_TRs_video_indices)

      train_run_TRs_video_indices = np.zeros(((num_towns -1)*num_TRs_video,), dtype=int)
      for i_train_run in range(num_towns-1):
          i_run_TRs_video_indices = copy.deepcopy(single_run_TRs_video_indices)
          np.random.shuffle(i_run_TRs_video_indices)
          train_run_TRs_video_indices[i_train_run*num_TRs_video:(i_train_run+1)*num_TRs_video] \
              = i_run_TRs_video_indices + i_train_run*num_TRs_video

      if i_shuffle%10 == 0:
          print("i_shuffle:",i_shuffle)

      i_Y_test_orig_permutation_list = []
      i_Y_test_pred_permutation_list = []

      for i_test_run in np.arange(num_towns):

          # print("i_run:",i_test_run)
          
          # get X data
          X_train = X_train_list[i_test_run]
          # X_train_permutation = X_train[train_run_TRs_video_indices[:,None], X_test_indices_2[None,:]]
          X_train_permutation = X_train

          X_test = X_test_list[i_test_run]
          # X_test_permutation = X_test[single_run_TRs_video_indices[:,None], X_test_indices_2[None,:]]
          X_test_permutation = X_test

          # get Y data
          Y_train = Y_train_list[i_test_run]
          Y_train_permutation = Y_train[train_run_TRs_video_indices[:,None], Y_test_indices_2[None,:]]

          Y_test = Y_test_list[i_test_run]
          Y_test_permutation = Y_test[single_run_TRs_video_indices[:,None], Y_test_indices_2[None,:]]

          # import
          from sklearn.model_selection import check_cv
          from voxelwise_tutorials.utils import generate_leave_one_run_out

          # indice of first sample of each run
          run_onsets = []
          num_run_train=7
          for i in range(num_run_train):
              run_onsets.append(i*num_TRs_video)
          # print(run_onsets)

          n_samples_train = X_train.shape[0]
          cv = generate_leave_one_run_out(n_samples_train, run_onsets)
          cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

          from sklearn.preprocessing import StandardScaler
          scaler = StandardScaler(with_mean=False, with_std=False)

          from voxelwise_tutorials.delayer import Delayer
          delayer = Delayer(delays=[0])

          from himalaya.kernel_ridge import KernelRidgeCV
          from himalaya.ridge import RidgeCV
          from himalaya.backend import set_backend
          backend = set_backend("torch_cuda", on_error="warn")
          # print(backend)

          X_train_permutation = X_train_permutation.astype("float32")
          X_test_permutation = X_test_permutation.astype("float32")

          alphas = np.logspace(-3, 20, 120)
          kernel_ridge_cv = RidgeCV(
              alphas=alphas, cv=cv, fit_intercept=True,
              solver_params=None) # try None

          from sklearn.pipeline import make_pipeline
          pipeline = make_pipeline(
              scaler,
              delayer,
              kernel_ridge_cv,
          )
          from sklearn import set_config
          set_config(display='diagram')  # requires scikit-learn 0.23
          _ = pipeline.fit(X_train_permutation, Y_train_permutation)

          # primal_coef = pipeline[-1].get_primal_coef()
          # primal_coef = backend.to_numpy(primal_coef)
          # print("(n_delays * n_features, n_voxels) =", primal_coef.shape)
          # print("coef mean:",np.mean(primal_coef.flatten()))

          # predict
          Y_test_permutation_predicted = pipeline.predict(X_test_permutation)
          i_Y_test_orig_permutation_list.append(Y_test_permutation)
          i_Y_test_pred_permutation_list.append(Y_test_permutation_predicted)

      # calc ev for each region in a permutation
      dim_accumulation = 0
      for i_region in range(0, len(brain_region_name_list)):

          dim_features = region_voxels_array[i_region]

          error = 0
          var = 0
          for i_test_run in np.arange(num_towns):

              Y_test = i_Y_test_orig_permutation_list[i_test_run]
              Y_test_predicted = i_Y_test_pred_permutation_list[i_test_run]

              y_true = Y_test[:, dim_accumulation:dim_accumulation + dim_features]
              y_pred = Y_test_predicted[:, dim_accumulation:dim_accumulation + dim_features]
              
              error = error + ((y_true - y_pred) ** 2.0).sum()
              var = var + ((y_true - y_true.mean(0)) ** 2.0).sum()

          sub_score[i_region][i_shuffle] = 1.0 - error / var

          dim_accumulation = dim_accumulation + dim_features

  explained_variance_each_shuffle_semantic = copy.deepcopy(np.asarray(sub_score))
  print("explained_variance_each_shuffle_semantic shape:", explained_variance_each_shuffle_semantic.shape)

  from scipy.io import savemat
  import copy

  print("explained_variance_each_shuffle_semantic shape:", explained_variance_each_shuffle_semantic.shape)

  matdic = {"explained_variance_each_shuffle_semantic": explained_variance_each_shuffle_semantic}
  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"

  file_dir = project_dir+"/process/shared_glm/temp_data/"
  savemat(file_dir+"vit_"+ str(num_total_dims) + "_dims"+ "_shareddims_explained_variance_each_shuffle_semantic.mat", matdic)

  ##### permuation explainable variance (each shuffle) - calc p-values
  print("explained_variance_each_shuffle_semantic.shape:",explained_variance_each_shuffle_semantic.shape)
  print("explained_variance_semantic.shape",explained_variance_semantic.shape)

  temp_shuffle_score = copy.deepcopy(explained_variance_each_shuffle_semantic)

  raw_pvalue_list = []
  for i_region in range(temp_shuffle_score.shape[0]):
    raw_pvalue = np.sum(temp_shuffle_score[i_region,:] > explained_variance_semantic[i_region])
    raw_pvalue_list.append(raw_pvalue)

  raw_pvalue_array = np.asarray(raw_pvalue_list)/temp_shuffle_score.shape[0]

  #-----multiple test
  import statsmodels

  # multiple test corrected for p-values
  rejects, pvals_corrected, _, _ = statsmodels.stats.multitest.multipletests(raw_pvalue_array)

  num_rejects = np.sum(rejects == True)
  print("num_rejects:", num_rejects)

  # plt.plot(pvals_corrected)
  print("pvals less than:", np.sum(pvals_corrected < 0.05))
  ##### permuation explainable variance (each shuffle) - visualize before multipletests
  semantic_sub_score_percentile = np.percentile(temp_shuffle_score, 95, axis=1)
  print('sub_score_shuffle_array shape:',temp_shuffle_score.shape)

  ##### permuation explainable variance (each shuffle) - save and load for wb
  from scipy.io import savemat
  import copy

  print("sub_score shape:", explained_variance_semantic.shape)

  if pvals_corrected is None:
    matdic = {"sub_score": explained_variance_semantic}
  else:
    matdic = {"sub_score": explained_variance_semantic,
              "pvals_corrected": pvals_corrected}
  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"

  file_dir = project_dir+"/process/shared_glm/temp_data/"
  savemat(file_dir+"vit_"+ str(num_total_dims) + "_dims"+ "_semantic_ev_score_brain_regions.mat", matdic)

  ### train the middle layer geometry model

  #### split training and testing data
  # take only one run for one town
  num_run = 14
  num_towns = 8

  Y_train_list = []
  Y_test_list = []

  # take one run as test
  for i_test_run in np.arange(num_towns):

    # spliting into train and test data
    Y_train_regions_list = []
    Y_test_regions_list = []
    Y_train_regions_array = np.empty((1400,0), float)
    Y_test_regions_array = np.empty((200,0), float)

    for i_region in range(0, len(semgeo_shared_region_glm_list)):

      temp_shared_response = semgeo_shared_region_glm_list[i_region]

      Y_data_temp = rearrange(temp_shared_response, 'i (j k) -> i j k', j=num_run)

      Y_data_array = np.concatenate((Y_data_temp[:,0:7,start_fixation_TRs+hemodynamic_shift_TRs:start_fixation_TRs+hemodynamic_shift_TRs+num_TRs_video], 
                Y_data_temp[:,num_run-1:num_run,start_fixation_TRs+hemodynamic_shift_TRs:start_fixation_TRs+hemodynamic_shift_TRs+num_TRs_video]), axis=1)

      Y_train = Y_data_array[:,np.arange(num_towns)!=i_test_run,:]
      Y_test = Y_data_array[:,np.arange(num_towns)==i_test_run,:]

      Y_train = rearrange(Y_train, 'i j k -> i (j k) ')
      Y_test = rearrange(Y_test, 'i j k -> i (j k) ')

      Y_train = Y_train.T
      Y_test = Y_test.T

      Y_train_regions_array = np.append(Y_train_regions_array, Y_train, axis=1)
      Y_test_regions_array = np.append(Y_test_regions_array, Y_test, axis=1)

    Y_train_list.append(Y_train_regions_array)
    Y_test_list.append(Y_test_regions_array)

  print("Y_train (n_TRs, shared_features) =", Y_train.shape)
  print("Y_test (n_TRs, shared_features) =", Y_test.shape)

  print("Y_train_regions_array (n_TRs, n_regions*shared_features) =", Y_train_regions_array.shape)
  print("Y_test_regions_array (n_TRs, n_regions*shared_features) =", Y_test_regions_array.shape)

  if(np.isnan(Y_train_regions_array).any()):
      print("Y_test_regions_array contain NaN values")

  if(np.isnan(Y_test_regions_array).any()):
      print("Y_test_regions_array contain NaN values")
  X_data_temp = []
  for i_run in np.arange(num_towns):
      data_temp = design_matrix_alltowns[223*i_run:223*(i_run+1),-(num_deeplayers_units+num_semantic_categories):] # first 8, last num_semantic_categories
      data_temp = data_temp[start_fixation_TRs+hemodynamic_shift_TRs:start_fixation_TRs+hemodynamic_shift_TRs+num_TRs_video, 0:num_midlayer_units]
      X_data_temp.append(data_temp)

  X_data_temp = np.asarray(X_data_temp)

  X_train_list = []
  X_test_list = []

  for i_test_run in np.arange(num_towns):

      X_train = X_data_temp[np.arange(num_towns)!=i_test_run,:,:]
      X_test = X_data_temp[np.arange(num_towns)==i_test_run,:,:]

      X_train = rearrange(X_train, 'i j k -> (i j) k')
      X_test = rearrange(X_test, 'i j k -> (i j) k')

      X_train_list.append(X_train)
      X_test_list.append(X_test)

      if(np.isnan(X_train).any()):
          print("X_train contain NaN values")

      if(np.isnan(X_test).any()):
          print("X_test contain NaN values")

  print("X_train (n_TRs, n_latent_units) =", X_train.shape)
  print("X_test (n_TRs, n_latent_units) =", X_test.shape)

  #### fit the model
  score_subs_list = []
  Y_test_orig_list = []
  Y_test_pred_list = []
  model_kernel_ridgecv_list = []

  for i_test_run in np.arange(num_towns):

      print("i_run:",i_test_run)
      
      # get X data
      X_train = X_train_list[i_test_run]
      X_test = X_test_list[i_test_run]

      # get Y data
      Y_train = Y_train_list[i_test_run]
      Y_test = Y_test_list[i_test_run]

      # import
      from sklearn.model_selection import check_cv
      from voxelwise_tutorials.utils import generate_leave_one_run_out

      # indice of first sample of each run
      run_onsets = []
      num_run_train=7
      for i in range(num_run_train):
          run_onsets.append(i*num_TRs_video)
      # print(run_onsets)

      n_samples_train = X_train.shape[0]
      cv = generate_leave_one_run_out(n_samples_train, run_onsets)
      cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler(with_mean=False, with_std=False)

      from voxelwise_tutorials.delayer import Delayer
      delayer = Delayer(delays=[0])

      from himalaya.kernel_ridge import KernelRidgeCV
      from himalaya.ridge import RidgeCV
      from himalaya.backend import set_backend
      backend = set_backend("torch_cuda", on_error="warn")
      # print(backend)

      X_train = X_train.astype("float32")
      X_test = X_test.astype("float32")

      alphas = np.logspace(0, 20, 120)
      kernel_ridge_cv = RidgeCV(
          alphas=alphas, cv=cv, fit_intercept=True,
          solver_params=None) # try None

      from sklearn.pipeline import make_pipeline
      pipeline = make_pipeline(
          scaler,
          delayer,
          kernel_ridge_cv,
      )
      from sklearn import set_config
      set_config(display='diagram')  # requires scikit-learn 0.23
      _ = pipeline.fit(X_train, Y_train)

      # test
      scores = pipeline.score(X_test, Y_test)
      scores = backend.to_numpy(scores)
      score_subs_list.append(scores)

      # predict
      Y_test_predicted = pipeline.predict(X_test)
      Y_test_orig_list.append(Y_test)
      Y_test_pred_list.append(Y_test_predicted)

      # append ridgecv
      model_kernel_ridgecv_list.append(pipeline)

      # plot best alphas
      if glm_show_alpha == True:
          from himalaya.viz import plot_alphas_diagnostic
          import matplotlib.pyplot as plt
          best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
          print('best_alphas:',best_alphas.shape)
          plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
          plt.show()

  score_subs_array = np.array(score_subs_list)
  print("score_subs_array.shape:",score_subs_array.shape)

  Y_test_orig_midlayer_geometry = np.asarray(Y_test_orig_list)
  print("Y_test_orig_midlayer_geometry.shape:",Y_test_orig_midlayer_geometry.shape)

  Y_test_pred_midlayer_geometry = np.asarray(Y_test_pred_list)
  print("Y_test_pred_midlayer_geometry.shape:",Y_test_pred_midlayer_geometry.shape)

  ##### explainable variance of dims
  from sklearn.metrics import r2_score

  # ve for each brain regions
  dim_accumulation = 0
  sub_score = np.zeros((Y_test_pred_midlayer_geometry.shape[2],), dtype=float)

  for i_dim in range(0, Y_test_pred_midlayer_geometry.shape[2]):

      error = 0
      var = 0
      for i_test_run in np.arange(num_towns):

          Y_test = Y_test_orig_list[i_test_run]
          Y_test_predicted = Y_test_pred_list[i_test_run]

          y_true = Y_test[:, i_dim:i_dim + 1]
          y_pred = Y_test_predicted[:, i_dim:i_dim + 1]
          
          error = error + ((y_true - y_pred) ** 2.0).sum()
          var = var + ((y_true - y_true.mean(0)) ** 2.0).sum()

      dim_accumulation = dim_accumulation + dim_features

      sub_score[i_dim] = 1.0 - error / var
      
  # show summary ev runs
  explained_variance_single_dim_midlayer_geometry = copy.deepcopy(np.asarray(sub_score))


  from scipy.io import savemat
  import copy

  print("explained_variance_single_dim_midlayer_geometry shape:", explained_variance_single_dim_midlayer_geometry.shape)

  matdic = {"explained_variance_single_dim_midlayer_geometry": explained_variance_single_dim_midlayer_geometry}
  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"

  file_dir = project_dir+"/process/shared_glm/temp_data/"
  savemat(file_dir+"vit_"+ str(num_total_dims) + "_dims"+ "_single_dim_midlayer_geometry_ev_score_brain_regions.mat", matdic)

  print("num of effective dims:", np.sum(explained_variance_single_dim_midlayer_geometry > 0.0))

  #### explainable variance with p-values 
  ##### ev on cross all runs
  from sklearn.metrics import r2_score

  # ve for each brain regions
  dim_accumulation = 0
  sub_score = np.zeros((len(brain_region_name_list),), dtype=float)

  for i_region in range(0, len(brain_region_name_list)):
      dim_features = region_voxels_array[i_region]

      error = 0
      var = 0
      for i_test_run in np.arange(num_towns):

          Y_test = Y_test_orig_list[i_test_run]
          Y_test_predicted = Y_test_pred_list[i_test_run]

          y_true = Y_test[:, dim_accumulation:dim_accumulation + dim_features]
          y_pred = Y_test_predicted[:, dim_accumulation:dim_accumulation + dim_features]
          
          error = error + ((y_true - y_pred) ** 2.0).sum()
          var = var + ((y_true - y_true.mean(0)) ** 2.0).sum()

      dim_accumulation = dim_accumulation + dim_features

      # each_sub_score[i_region] = r2_score(y_true, y_pred)
      # error = ((y_true - y_pred) ** 2.0).sum()
      # var = ((y_true - y_true.mean(0)) ** 2.0).sum()
      sub_score[i_region] = 1.0 - error / var
      
  # show summary ev runs
  explained_variance_midlayer_geometry = copy.deepcopy(np.asarray(sub_score))

  #### permutation before cross-validation

  ##### the geometry permutation model
  # ve for each brain regions
  shuffle_num_times = 10
  sub_score_shuffle_list = []

  Y_test_orig_permutation_list = []
  Y_test_pred_permutation_list = []

  sub_score = np.zeros((len(brain_region_name_list), shuffle_num_times,), dtype=float)

  for i_shuffle in range(shuffle_num_times):

      np.random.seed(i_shuffle)
      single_run_TRs_video_indices = np.arange(num_TRs_video)
      X_test_indices_2 = np.arange(X_test_list[0].shape[1])
      Y_test_indices_2 = np.arange(Y_test_list[0].shape[1])
      np.random.shuffle(single_run_TRs_video_indices)

      train_run_TRs_video_indices = np.zeros(((num_towns - 1)*num_TRs_video,), dtype=int)
      for i_train_run in range(num_towns-1):
          i_run_TRs_video_indices = copy.deepcopy(single_run_TRs_video_indices)
          np.random.shuffle(i_run_TRs_video_indices)
          train_run_TRs_video_indices[i_train_run*num_TRs_video:(i_train_run+1)*num_TRs_video] \
              = i_run_TRs_video_indices + i_train_run*num_TRs_video

      if i_shuffle%10 == 0:
          print("i_shuffle:",i_shuffle)

      i_Y_test_orig_permutation_list = []
      i_Y_test_pred_permutation_list = []

      for i_test_run in np.arange(num_towns):

          # print("i_run:",i_test_run)
          
          # get X data
          X_train = X_train_list[i_test_run]
          # X_train_permutation = X_train[train_run_TRs_video_indices[:,None], X_test_indices_2[None,:]]
          X_train_permutation = X_train

          X_test = X_test_list[i_test_run]
          # X_test_permutation = X_test[single_run_TRs_video_indices[:,None], X_test_indices_2[None,:]]
          X_test_permutation = X_test

          # get Y data
          Y_train = Y_train_list[i_test_run]
          Y_train_permutation = Y_train[train_run_TRs_video_indices[:,None], Y_test_indices_2[None,:]]

          Y_test = Y_test_list[i_test_run]
          Y_test_permutation = Y_test[single_run_TRs_video_indices[:,None], Y_test_indices_2[None,:]]

          # import
          from sklearn.model_selection import check_cv
          from voxelwise_tutorials.utils import generate_leave_one_run_out

          # indice of first sample of each run
          run_onsets = []
          num_run_train=7
          for i in range(num_run_train):
              run_onsets.append(i*num_TRs_video)
          # print(run_onsets)

          n_samples_train = X_train.shape[0]
          cv = generate_leave_one_run_out(n_samples_train, run_onsets)
          cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

          from sklearn.preprocessing import StandardScaler
          scaler = StandardScaler(with_mean=False, with_std=False)

          from voxelwise_tutorials.delayer import Delayer
          delayer = Delayer(delays=[0])

          from himalaya.kernel_ridge import KernelRidgeCV
          from himalaya.ridge import RidgeCV
          from himalaya.backend import set_backend
          backend = set_backend("torch_cuda", on_error="warn")
          # print(backend)

          X_train_permutation = X_train_permutation.astype("float32")
          X_test_permutation = X_test_permutation.astype("float32")

          alphas = np.logspace(-3, 20, 120)
          kernel_ridge_cv = RidgeCV(
              alphas=alphas, cv=cv, fit_intercept=True,
              solver_params=None) # try None

          from sklearn.pipeline import make_pipeline
          pipeline = make_pipeline(
              scaler,
              delayer,
              kernel_ridge_cv,
          )
          from sklearn import set_config
          set_config(display='diagram')  # requires scikit-learn 0.23
          _ = pipeline.fit(X_train_permutation, Y_train_permutation)

          # primal_coef = pipeline[-1].get_primal_coef()
          # primal_coef = backend.to_numpy(primal_coef)
          # print("(n_delays * n_features, n_voxels) =", primal_coef.shape)
          # print("coef mean:",np.mean(primal_coef.flatten()))

          # predict
          Y_test_permutation_predicted = pipeline.predict(X_test_permutation)
          i_Y_test_orig_permutation_list.append(Y_test_permutation)
          i_Y_test_pred_permutation_list.append(Y_test_permutation_predicted)

      # calc ev for each region in a permutation
      dim_accumulation = 0
      for i_region in range(0, len(brain_region_name_list)):

          dim_features = region_voxels_array[i_region]

          error = 0
          var = 0
          for i_test_run in np.arange(num_towns):

              Y_test = i_Y_test_orig_permutation_list[i_test_run]
              Y_test_predicted = i_Y_test_pred_permutation_list[i_test_run]

              y_true = Y_test[:, dim_accumulation:dim_accumulation + dim_features]
              y_pred = Y_test_predicted[:, dim_accumulation:dim_accumulation + dim_features]
              
              error = error + ((y_true - y_pred) ** 2.0).sum()
              var = var + ((y_true - y_true.mean(0)) ** 2.0).sum()

          sub_score[i_region][i_shuffle] = 1.0 - error / var

          dim_accumulation = dim_accumulation + dim_features

  explained_variance_each_shuffle_midlayer_geometry = copy.deepcopy(np.asarray(sub_score))
  print("explained_variance_each_shuffle_midlayer_geometry shape:", explained_variance_each_shuffle_midlayer_geometry.shape)

  from scipy.io import savemat
  import copy

  print("explained_variance_each_shuffle_midlayer_geometry shape:", explained_variance_each_shuffle_midlayer_geometry.shape)

  matdic = {"explained_variance_each_shuffle_midlayer_geometry": explained_variance_each_shuffle_midlayer_geometry}
  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"

  file_dir = project_dir+"/process/shared_glm/temp_data/"
  savemat(file_dir+"vit_"+ str(num_total_dims) + "_dims"+ "_shareddims_explained_variance_each_shuffle_midlayer_geometry.mat", matdic)

  from scipy.io import loadmat

  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"
  file_dir = project_dir+"/process/shared_glm/temp_data/"
  matdic = loadmat(file_dir+"vit_"+ str(num_total_dims) + "_dims"+ "_shareddims_explained_variance_each_shuffle_midlayer_geometry.mat")

  explained_variance_each_shuffle_midlayer_geometry = matdic["explained_variance_each_shuffle_midlayer_geometry"]
  print("explained_variance_each_shuffle_midlayer_geometry shape =", explained_variance_each_shuffle_midlayer_geometry.shape)
  ##### permuation explainable variance (each shuffle) - calc p-values
  print("explained_variance_each_shuffle_midlayer_geometry.shape:",explained_variance_each_shuffle_midlayer_geometry.shape)
  print("explained_variance_midlayer_geometry.shape",explained_variance_midlayer_geometry.shape)

  temp_shuffle_score =  copy.deepcopy(explained_variance_each_shuffle_midlayer_geometry.T)

  raw_pvalue_list = []
  for i_region in range(temp_shuffle_score.shape[1]):
    raw_pvalue = np.sum(temp_shuffle_score[:,i_region] > explained_variance_midlayer_geometry[i_region])
    raw_pvalue_list.append(raw_pvalue)

  raw_pvalue_array = np.asarray(raw_pvalue_list)/temp_shuffle_score.shape[0]

  #-----multiple test
  import statsmodels

  # multiple test corrected for p-values
  rejects, pvals_corrected, _, _ = statsmodels.stats.multitest.multipletests(raw_pvalue_array)

  num_rejects = np.sum(rejects == True)
  print("num_rejects:", num_rejects)

  # plt.plot(pvals_corrected)
  print("pvals less than:", np.sum(pvals_corrected < 0.05))
  ##### permuation explainable variance (each shuffle) - visualize before multipletests
  midlayer_geometry_sub_score_percentile = np.percentile(temp_shuffle_score, 95, axis=0)
  print('sub_score_shuffle_array shape:',temp_shuffle_score.shape)

  ##### permuation explainable variance (each shuffle) - save and load for wb
  from scipy.io import savemat
  import copy

  print("sub_score shape:", explained_variance_midlayer_geometry.shape)

  if pvals_corrected is None:
    matdic = {"sub_score": explained_variance_midlayer_geometry}
  else:
    matdic = {"sub_score": explained_variance_midlayer_geometry,
              "pvals_corrected": pvals_corrected}
  project_dir="/media/statespace/Spatial/sptialworkspace/spatialfMRI/fMRI_analysis/igeo_process"

  file_dir = project_dir+"/process/shared_glm/temp_data/"
  savemat(file_dir+"vit_"+ str(num_total_dims) + "_dims"+ "_midlayer_geometry_ev_score_brain_regions.mat", matdic)

