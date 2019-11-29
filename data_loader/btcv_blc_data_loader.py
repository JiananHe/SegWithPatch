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
from time import time
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from utils import setup_seed, organs_properties

sample_size = 64  # 64*64*64 for a sample
num_organ = organs_properties['num_organ']
organs_name = organs_properties['organs_name']

setup_seed(time())


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


def weight_to_numbers(n_samples, organs_weight):
    n_per_class = np.round(n_samples * (np.array(organs_weight) / np.sum(organs_weight)))
    s = " ".join(["%s:%.3f" % (i, j) for i, j in zip(organs_name, n_per_class)])
    os.system('echo %s' % "samples every organ:")
    os.system('echo %s' % s)
    return n_per_class


class MyDataset(Dataset):
    def __init__(self, csv_path, n_samples, organs_weight):
        """
        Data loader. note that the loader return n_samples extracted from one volume in one batch.
        When use dataloader, batch size need to be 1.
        :param csv_path: csv file contains the vol path
        :param n_vols: vol number
        :param n_samples: samples number in one batch
        """
        assert n_samples >= num_organ
        if 'train' in csv_path:
            self.is_training = True
        else:
            self.is_training = False

        self.n_samples = n_samples
        self.organs_weight = organs_weight

        csv_reader = csv.reader(open(csv_path, 'r'))
        self.vol_info = [row for row in csv_reader]

        # 根据weight计算各个器官的采样patch数目
        self.n_per_class = weight_to_numbers(self.n_samples, self.organs_weight)

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
        vol_shape = img_array.shape

        # 根据n_per_class选取sample
        img_samples = []
        lbl_samples = []
        t = int(sample_size / 2)
        for cls_idx in range(num_organ):
            organ_points = np.argwhere(lbl_array == cls_idx+1)
            if self.n_per_class[cls_idx] == 0 or len(organ_points) == 0:
                continue

            np.random.shuffle(organ_points)
            for pc, point in enumerate(organ_points):
                # shift to avoid crossing border
                point = [point[i] if point[i] <= s-t else s-t for i, s in enumerate(vol_shape)]
                point = [point[i] if point[i] >= t else t for i, s in enumerate(vol_shape)]

                img_samples.append(img_array[
                                   point[0] - t:point[0] + t,
                                   point[1] - t:point[1] + t,
                                   point[2] - t:point[2] + t])
                lbl_samples.append(lbl_array[
                                   point[0] - t:point[0] + t,
                                   point[1] - t:point[1] + t,
                                   point[2] - t:point[2] + t])
                if pc == self.n_per_class[cls_idx]-1:
                    break

        samples_array = np.array(img_samples).astype(np.float32)  # (n_samples, sample_size, sample_size, sample_size)
        samples_label = np.array(lbl_samples).astype(np.uint8)  # (n_samples, sample_size, sample_size, sample_size)

        # shuffle samples
        samples_idx = list(range(samples_array.shape[0]))
        np.random.shuffle(samples_idx)
        samples_array = np.array([samples_array[i] for i in samples_idx])
        samples_label = np.array([samples_label[i] for i in samples_idx])

        # 处理完毕，将array转换为tensor
        samples_array = torch.FloatTensor(samples_array).unsqueeze(1)  # (n_samples, 1, sample_size, sample_size, sample_size)
        samples_label = torch.LongTensor(samples_label)

        return samples_array, samples_label


if __name__ == "__main__":
    while(True):
        train_ds = MyDataset(r'..\csv_files\btcv_train_info.csv', 24, organs_weight=organs_properties['organs_weight'])
        train_dl = DataLoader(train_ds, 1, True)

        for index, (ct, seg) in enumerate(train_dl):
            ct = ct.squeeze(0)
            seg = seg.squeeze(0)
            print(index, ct.size(), seg.size())
            print((seg.numpy()==0).sum())

            print('----------------')


