import numpy as np
import pandas as pd
from skimage import io

from tqdm import tqdm_notebook
from torch.utils import data
from skimage.transform import resize

from sklearn.model_selection import train_test_split


class SaltLoader(data.Dataset):
    def __init__(self, root, split="train",img_size_ori=101,img_size_target=128,
                 img_norm=False,split_size=0.2):
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
        print('Getting and resizing train images and masks ... ')

        if split!="test":
            train_df["images"] = [
                np.array(io.imread(self.root + "train/images/{}.png".format(idx), as_grey=True), dtype=np.float32) for
                idx
                in tqdm_notebook(train_df.index)]
            train_df["masks"] = [
                np.array(io.imread(self.root + "train/masks/{}.png".format(idx), as_grey=True), dtype=np.bool_) for idx
                in
                tqdm_notebook(train_df.index)]

            train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
            train_df["coverage_class"] = train_df.coverage.map(self.cov_to_class)

            # split data
            self.ids_train, self.ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
                train_df.index.values,
                np.array(train_df.images.map(self.upsample).tolist(), dtype=np.float32).reshape(-1, 1, img_size_target,
                                                                                                img_size_target),
                np.array(train_df.masks.map(self.upsample).tolist(), dtype=np.float32).astype(np.float32).reshape(-1, 1,
                                                                                                                  img_size_target,
                                                                                                                  img_size_target),
                train_df.coverage.values,
                train_df.z.values,
                test_size=split_size, stratify=train_df.coverage_class, random_state=1337)

            # flip images
            x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
            y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

            self.train_df = train_df

            if split == "train":
                self.images = x_train
                self.masks = y_train
            elif split == "val":
                self.images = x_valid
                self.masks = y_valid
        else:
            test_df["images"]=[
                np.array(io.imread(self.root + "images/{}.png".format(idx), as_grey=True), dtype=np.float32) for
                idx
                in tqdm_notebook(test_df.index)]
            self.images=np.array(test_df.images.map(self.upsample).tolist(), dtype=np.float32).reshape(-1, 1, img_size_target,
                                                                                                img_size_target)
            self.test_df = test_df


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.split!="test":
            mask = self.masks[idx]
            return (image, mask)
        else:
            return image


    def cov_to_class(self,val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i

    def upsample(self,img):
        if self.img_size_ori == self.img_size_target:
            return img
        return resize(img, (self.img_size_target, self.img_size_target), mode='constant', preserve_range=True)
