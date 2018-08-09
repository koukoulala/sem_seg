import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from keras.preprocessing.image import load_img
from tqdm import tqdm_notebook
from torch.utils import data


class TGSLoader(data.Dataset):
    def __init__(self, root, img_size_ori=101,img_size_target=128, img_norm=False):
        self.root = root
        self.img_norm = img_norm
        self.img_size_ori=img_size_ori
        self.img_size_target=img_size_target

        #data loading
        self.train_df = pd.read_csv(self.root+"train.csv", index_col="id", usecols=[0])
        depths_df = pd.read_csv(self.root+"depths.csv", index_col="id")
        self.train_df = self.train_df.join(depths_df)
        self.test_df = depths_df[~depths_df.index.isin(self.train_df.index)]

        self.train_df["images"] = [
            np.array(load_img(self.root+"train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in
            tqdm_notebook(self.train_df.index)]
        self.train_df["masks"] = [
            np.array(load_img(self.root+"train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in
            tqdm_notebook(self.train_df.index)]

        self.train_df["coverage"] = self.train_df.masks.map(np.sum) / pow(img_size_ori, 2)
        self.train_df["coverage_class"] = self.train_df.coverage.map(self.cov_to_class)

    def cov_to_class(self,val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i
