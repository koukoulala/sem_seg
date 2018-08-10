import numpy as np
import pandas as pd
from skimage import io

from tqdm import tqdm_notebook
from torch.utils import data
from skimage.transform import resize

from sklearn.model_selection import train_test_split


class SaltLoader(data.Dataset):
    def __init__(self, root, split="train",img_size_ori=101,img_size_target=128,
                 img_norm=False,channels=1):
        self.root = root
        self.split=split
        self.img_norm = img_norm
        self.img_size_ori=img_size_ori
        self.img_size_target=img_size_target

        #data loading
        train_df = pd.read_csv(self.root+"train.csv", index_col="id", usecols=[0])
        depths_df = pd.read_csv(self.root+"depths.csv", index_col="id")
        train_df = train_df.join(depths_df)
        test_df = depths_df[~depths_df.index.isin(train_df.index)]

        # Get and resize train images and masks
        X_train = np.zeros((len(train_df), img_size_target, img_size_target, channels), dtype=np.uint8)
        Y_train = np.zeros((len(train_df), img_size_target, img_size_target, 1), dtype=np.bool_)
        print('Getting and resizing train images and masks ... ')

        self.train_df["images"] = [
            np.array(io.imread(self.root + "train/images/{}.png".format(idx), as_grey=True), dtype=np.float32) for idx
            in tqdm_notebook(self.train_df.index)]
        self.train_df["masks"] = [
            np.array(io.imread(self.root + "train/masks/{}.png".format(idx), as_grey=True), dtype=np.bool_) for idx in
            tqdm_notebook(self.train_df.index)]

        self.train_df["coverage"] = self.train_df.masks.map(np.sum) / pow(img_size_ori, 2)
        self.train_df["coverage_class"] = self.train_df.coverage.map(self.cov_to_class)
        '''
        for n,idx in tqdm_notebook(enumerate(train_df.index),total=len(train_df.index)):
            img = io.imread(self.root+"train/images/{}.png".format(idx))
            x = resize(img, (128, 128, 1), mode='constant', preserve_range=True)
            X_train[n]=x
            #train_df["images"]=np.array(x,dtype=np.uint8)
            mask = io.imread(self.root+"train/masks/{}.png".format(idx))
            y=resize(mask, (128, 128, 1),mode='constant',preserve_range=True)
            Y_train[n]=y
            #train_df["masks"]=np.array(y,dtype=np.bool_)

        X_train_shaped = X_train.reshape(-1, 1, 128, 128) / 255
        Y_train_shaped = Y_train.reshape(-1, 1, 128, 128)
        X_train_shaped = X_train_shaped.astype(np.float32)
        Y_train_shaped = Y_train_shaped.astype(np.float32)
        '''

        # split data
        ids_train, ids_valid, x_train, x_valid, y_train, y_valid, depth_train, depth_test = train_test_split(
            train_df.index.values,
            np.array(train_df.images.map(self.upsample).tolist(),dtype=np.float32).reshape(-1, img_size_target, img_size_target, 1),
            np.array(train_df.masks.map(self.upsample).tolist(),dtype=np.float32).reshape(-1, img_size_target, img_size_target, 1),
            train_df.z.values,
            test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

        if split=="train":
            self.images=x_train
            self.masks=y_train
        elif split=="val":
            self.images=x_valid
            self.masks=y_valid
        else:
            print("not done")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = None
        if self.train!="test":
            mask = self.masks[idx]
        return (image, mask)


    def cov_to_class(self,val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i

    def upsample(self,img):
        if self.img_size_ori == self.img_size_target:
            return img
        return resize(img, (self.img_size_target, self.img_size_target), mode='constant', preserve_range=True)
