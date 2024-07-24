"""Utility functions."""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from torch import nn
import PIL.Image
from scipy.io import savemat, loadmat
from datetime import datetime
# from visdom import Tensor, Visdom
import glob
import pickle
import random
from joblib import Parallel, delayed

# Define data loading step
class dataloader_rgb_depth(Dataset):

    def __init__(self, train_or_test, root_dir, start_index, run_num, town_num, run_sample_size, sample_step, batch_size, device=None):
        self.root_dir = root_dir

        self.run_num = run_num
        self.town_num = town_num
        self.run_sample_size = run_sample_size
        self.sample_step = sample_step
        self.start_index = start_index
        self.train_or_test = train_or_test
        self.batch_size = batch_size
        self.device = device
        
        self.pil_to_tensor = torch.nn.Sequential(
            torchvision.transforms.Resize([320, 640]),                   
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        )
        # ).to(self.device)
        # self.pil_to_tensor = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize([320, 640]),                   
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        #         std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        # ])

        self.depth_to_tensor = torch.nn.Sequential(
            torchvision.transforms.Resize([320, 640]),                   
            # torchvision.transforms.ToTensor(),
        )
        # ).to(self.device)
        # self.depth_to_tensor = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize([320, 640]),                   
        #     torchvision.transforms.ToTensor(),
        # ])

        self.towns_dict = {
            0: "Town01",
            1: "Town02",
            2: "Town03",
            3: "Town04",
            4: "Town05",
            5: "Town06",
            6: "Town07",
            7: "Town08",
            }

        self.weather_dict = {
            0: "ClearNight",
            1: "WetSunset",
            2: "WetCloudyNoon",
            3: "ClearNoon",
            4: "ClearSunset",
            5: "HardRainSunset",
            6: "FogMorning",
            }

        print("---Summary---")
        print("num of towns:", self.town_num)
        print("num of run in each town:", self.run_num)
        print("sample size of each town:", self.run_sample_size)
        print("sample step:", self.sample_step)

        self.num_images = int(self.town_num*self.run_num*self.run_sample_size/self.sample_step)
        print("num of totoal images:", self.num_images)

        self.indices = np.arange(self.num_images)
        if self.train_or_test == "train":
          np.random.shuffle(self.indices)

        self.batch_num = self.num_images//self.batch_size
        print("batch_size:", self.batch_size)
        print("batch_num:", self.batch_num)
        print("-----------")

        return

    def images_sample(self, batch_index):

        self.rgb_images = torch.zeros((self.batch_size, 3, 320, 640), device=self.device)
        self.depth_images = torch.zeros((self.batch_size, 320, 640), device=self.device)

        try:
            for i in range(self.batch_size):
                
                # calc index
                index_image = batch_index*self.batch_size + i
                index_image = self.indices[index_image]
                
                # shuffled index
                town_index = np.floor(index_image/self.run_sample_size/self.run_num).astype(int)
                run_index = np.floor(index_image/self.run_sample_size%self.run_num).astype(int)
                sample_index = np.floor(index_image%self.run_sample_size).astype(int)
                
                # find rgb path
                self.rgb_image_folder_str = self.towns_dict[town_index] + '_' + self.weather_dict[run_index]
                current_rgb_str = 'rgb_{idx:06}.jpg'
                self.current_wildcard_rgb_name = os.path.join(self.root_dir, self.rgb_image_folder_str, \
                        current_rgb_str.format(idx = sample_index + self.start_index))

                # read rgb image
                current_rbg_name = glob.glob(self.current_wildcard_rgb_name)[0]
                # print("current_rbg_name:",current_rbg_name)
                rgb_image = PIL.Image.open(current_rbg_name).convert('RGB')

                # random flip left right
                if self.train_or_test == "train":
                    bool_flip_left_right = random.choice([True, False])
                elif self.train_or_test == "test":
                    bool_flip_left_right = False

                if bool_flip_left_right == True:
                    rgb_image = rgb_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                rgb_img_data = self.pil_to_tensor(torchvision.transforms.functional.to_tensor(rgb_image))
                self.rgb_images[i] = rgb_img_data

                # find depth path'
                self.depth_image_folder_str = self.towns_dict[town_index] + '_' + self.weather_dict[run_index]
                current_depth_str = 'depth_{idx:06}.jpg'
                if self.train_or_test == "train":
                  self.current_wildcard_depth_name = os.path.join(self.root_dir, self.depth_image_folder_str, \
                          current_depth_str.format(idx = sample_index + self.start_index))
                elif self.train_or_test == "test":
                  self.depth_image_folder_str = self.towns_dict[town_index] + '_' + 'Depth'
                  self.current_wildcard_depth_name = os.path.join("/media/statespace/S/recording/output_synchronized", self.depth_image_folder_str, \
                          current_depth_str.format(idx = sample_index + self.start_index))

                # read depth image
                current_depth_name = glob.glob(self.current_wildcard_depth_name)[0]
                # print("current_depth_name:",current_depth_name)
                depth_image = PIL.Image.open(current_depth_name)
                if bool_flip_left_right == True:
                    depth_image = depth_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                depth_img_data = self.depth_to_tensor(torchvision.transforms.functional.to_tensor(depth_image))

                self.depth_images[i] = depth_img_data

                # print(current_depth_name)
                # print(current_rbg_name)

        finally:
            pass
            # print("batch load finished...")

        return self.rgb_images, self.depth_images


    def images_parallel_read(self, i, batch_index, rgb_images, depth_images):

        # calc index
        index_image = batch_index*self.batch_size + i
        index_image = self.indices[index_image]
        
        # shuffled index
        town_index = np.floor(index_image/self.run_sample_size/self.run_num).astype(int)
        run_index = np.floor(index_image/self.run_sample_size%self.run_num).astype(int)
        sample_index = np.floor(index_image%self.run_sample_size).astype(int)
        
        # find rgb path
        self.rgb_image_folder_str = self.towns_dict[town_index] + '_' + self.weather_dict[run_index]
        current_rgb_str = 'rgb_{idx:06}.jpg'
        self.current_wildcard_rgb_name = os.path.join(self.root_dir, self.rgb_image_folder_str, \
                current_rgb_str.format(idx = sample_index + self.start_index))

        # read rgb image
        current_rbg_name = glob.glob(self.current_wildcard_rgb_name)[0]
        # print("current_rbg_name:",current_rbg_name)
        rgb_image = PIL.Image.open(current_rbg_name).convert('RGB')

        # random flip left right
        if self.train_or_test == "train":
            bool_flip_left_right = random.choice([True, False])
        elif self.train_or_test == "test":
            bool_flip_left_right = False

        if bool_flip_left_right == True:
            rgb_image = rgb_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        rgb_img_data = self.pil_to_tensor(torchvision.transforms.functional.to_tensor(rgb_image))
        rgb_images[i] = rgb_img_data

        # find depth path'
        self.depth_image_folder_str = self.towns_dict[town_index] + '_' + self.weather_dict[run_index]
        current_depth_str = 'depth_{idx:06}.jpg'
        if self.train_or_test == "train":
            self.current_wildcard_depth_name = os.path.join(self.root_dir, self.depth_image_folder_str, \
                    current_depth_str.format(idx = sample_index + self.start_index))
        elif self.train_or_test == "test":
            self.depth_image_folder_str = self.towns_dict[town_index] + '_' + 'Depth'
            self.current_wildcard_depth_name = os.path.join("/media/statespace/S/recording/output_synchronized", self.depth_image_folder_str, \
                    current_depth_str.format(idx = sample_index + self.start_index))

        # read depth image
        current_depth_name = glob.glob(self.current_wildcard_depth_name)[0]
        # print("current_depth_name:",current_depth_name)
        depth_image = PIL.Image.open(current_depth_name)
        if bool_flip_left_right == True:
            depth_image = depth_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        depth_img_data = self.depth_to_tensor(torchvision.transforms.functional.to_tensor(depth_image))

        depth_images[i] = depth_img_data

        return

    def images_parallel_sample(self, batch_index, n_jobs):

        self.rgb_images = torch.zeros((self.batch_size, 3, 320, 640))
        self.depth_images = torch.zeros((self.batch_size, 320, 640))

        Parallel(n_jobs=n_jobs, require='sharedmem')(delayed(self.images_parallel_read)(i, batch_index, self.rgb_images, self.depth_images) for i in range(self.batch_size))

        return self.rgb_images, self.depth_images

class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, viz, env_name='main'):
        self.viz = viz
        self.env = env_name
        self.plots = {}

    def plot(self, var_name="loss", split_name="loss", title_name="loss", x = [], y = []):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=x, Y=y, env=self.env, name=split_name, opts=dict(
                legend=split_name,
                title=title_name,
                xlabel='ith iterations',
                ylabel=var_name,
                height=320
            ))
        else:
            self.viz.line(X=x, Y=y, env=self.env, win=self.plots[var_name], name=split_name, #update = 'append',
                opts=dict(
                    legend=split_name,
                    title=title_name,
                    xlabel='ith iterations',
                    ylabel=var_name,
                    height=320
                )
            )

    def multiplot(self, var_name="multiloss", x = [], y = []):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=x, Y=y, env=self.env,
            opts=dict(
                    title='loss',
                    xlabel='ith iterations',
                    ylabel='loss',
                    height=320,
                    ytickmin=0,
                    dash = np.array(['solid', 'dash', 'dashdot', 'solid', 'dash', 'dash']),
                    linecolor = np.array([
                        [0, 255, 191],
                        [0, 191, 255],
                        [255, 0, 0],
                        [204, 204, 0],
                        [102, 0, 204],
                        [204, 153, 255],
                    ]),
                    legend=['total_loss', 'recon_loss', 'kld_loss', 'total_kld', 'klds_var', 'corr_triu_loss'],
                )
            )
        else:
            self.viz.line(X=x, Y=y, env=self.env, win=self.plots[var_name],
            opts=dict(
                    title='loss',
                    xlabel='ith iterations',
                    ylabel='loss',
                    height=320,
                    ytickmin=0,
                    dash = np.array(['solid', 'dash', 'dashdot', 'solid', 'dash', 'dash']),
                    linecolor = np.array([
                        [0, 255, 191],
                        [0, 191, 255],
                        [255, 0, 0],
                        [204, 204, 0],
                        [102, 0, 204],
                        [204, 153, 255],
                    ]),
                    legend=['total_loss', 'recon_loss', 'kld_loss', 'total_kld', 'klds_var', 'corr_triu_loss'],
                )
            )

    def rgb_depth_images(self, var_name="depth", split_name="depth", title_name="depth", images = []):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env, 
                    opts=dict(title=title_name, height=300), nrow=3)

        else:
            self.viz.images(images, env=self.env, 
                    win=self.plots[var_name], 
                    opts=dict(title=title_name, height=300), nrow=3)






