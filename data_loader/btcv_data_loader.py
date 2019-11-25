# 数据增强
# 1. 随机旋转
# 2. 放大
# 3. 塑性形变

import torch
import SimpleITK as sitk
import scipy.ndimage as ndimage
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os
import csv
import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from utils import setup_seed, organs_properties

sample_size = 64  # 64*64*64 for a sample

setup_seed(2018)


def elastic_transform(image, alpha, sigma):
    """
    Elastic deformation of images as described in [1].

    [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        Neural Networks applied to Visual Document Analysis", in Proc. of the
        International Conference on Document Analysis and Recognition, 2003.

    Based on gist https://gist.github.com/erniejunior/601cdf56d2b424757de5

    Args:
        image (np.ndarray): image to be deformed
        alpha (list): scale of transformation for each dimension, where larger
            values have more deformation
        sigma (list): Gaussian window of deformation for each dimension, where
            smaller values have more localised deformation

    Returns:
        np.ndarray: deformed image
    """

    assert len(alpha) == len(sigma), \
        "Dimensions of alpha and sigma are different"

    channelbool = image.ndim - len(alpha)
    out = np.zeros((len(alpha) + channelbool, ) + image.shape)

    # Generate a Gaussian filter, leaving channel dimensions zeroes
    for jj in range(len(alpha)):
        array = (np.random.rand(*image.shape) * 2 - 1)
        out[jj] = gaussian_filter(array, sigma[jj],
                                  mode="constant", cval=0) * alpha[jj]

    # Map mask to indices
    shapes = list(map(lambda x: slice(0, x, None), image.shape))
    grid = np.broadcast_arrays(*np.ogrid[shapes])
    indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))

    # Transform image based on masked indices
    transformed_image = map_coordinates(image, indices, order=0,
                                        mode='reflect').reshape(image.shape)

    return transformed_image


class MyDataset(Dataset):
    def __init__(self, csv_path, n_samples):
        """
        Data loader. note that the loader return n_samples extracted from one volume in one batch.
        When use dataloader, batch size need to be 1.
        :param csv_path: csv file contains the vol path
        :param n_vols: vol number
        :param n_samples: samples number in one batch
        """
        if 'train' in csv_path:
            self.is_training = True
        else:
            self.is_training = False

        self.n_samples = n_samples

        csv_reader = csv.reader(open(csv_path, 'r'))
        self.vol_info = [row for row in csv_reader]

    def __len__(self):
        random.shuffle(self.vol_info)
        return len(self.vol_info)

    def __getitem__(self, item):
        img_path = self.vol_info[item][0]
        lbl_path = self.vol_info[item][1]

        # 读取volume
        image = sitk.ReadImage(img_path)
        label = sitk.ReadImage(lbl_path)
        img_array = sitk.GetArrayFromImage(image)
        lbl_array = sitk.GetArrayFromImage(label)
        rank = img_array.ndim
        s = img_array.shape

        # 随机选取n_samples个samples
        img_samples = []
        lbl_samples = []
        organs_points = np.argwhere(lbl_array != 0)
        np.random.shuffle(organs_points)

        t = int(sample_size/2)
        valid_points = filter(lambda p: np.all([t <= p[i] <= s[i] - t for i in range(rank)]), organs_points)

        # random_loc = [np.random.randint(0, valid_loc[i], size=self.n_samples) for i in range(rank)]  # shape: [3, n_samples]
        for i, point in enumerate(valid_points):
            img_samples.append(img_array[
                         point[0] - t:point[0] + t,
                         point[1] - t:point[1] + t,
                         point[2] - t:point[2] + t])
            lbl_samples.append(lbl_array[
                         point[0] - t:point[0] + t,
                         point[1] - t:point[1] + t,
                         point[2] - t:point[2] + t])
            if i == self.n_samples-1:
                break

        samples_array = np.array(img_samples).astype(np.float32)  # (n_samples, sample_size, sample_size, sample_size)
        samples_label = np.array(lbl_samples).astype(np.uint8)  # (n_samples, sample_size, sample_size, sample_size)

        # 处理完毕，将array转换为tensor
        samples_array = torch.FloatTensor(samples_array).unsqueeze(1)  # (n_samples, 1, sample_size, sample_size, sample_size)
        samples_label = torch.LongTensor(samples_label)

        return samples_array, samples_label



if __name__ == "__main__":
    train_ds = MyDataset(r'..\csv_files\btcv_train_info.csv', 5)

    # 测试代码
    train_dl = DataLoader(train_ds, 1, True)
    while(True):
        for index, (ct, seg) in enumerate(train_dl):
            ct = ct.squeeze(0)
            seg = seg.squeeze(0)
            print(index, ct.size(), seg.size())
            seg_arr = seg.numpy()
            for i in range(seg_arr.shape[0]):
                bg_count = (seg_arr[i] == 0).sum()
                print(bg_count/np.product(seg_arr[i].shape))
            print('----------------')


