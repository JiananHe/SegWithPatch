# 数据预处理：
# 1. 灰度值阶段[-125, 275]
# 2. 裁剪掉零值区域
# 3. 归一化（每个样本单独归一化，-1~1）
# 4. 重采样（new spacing为median spacing）[0.758 0.758 3.   ]

import SimpleITK as sitk
import os
import numpy as np
import shutil
import scipy.ndimage as ndimage
import csv
import random
from skimage import measure


random.seed(123)

# ct_upper = 275.0
# ct_lower = -125.0
ct_upper = 350.0
ct_lower = -350.0
new_spacing = [0.758, 0.758, 3.0]


def preprocess(img_name, img_vol, lbl_vol=None):
    img_array = sitk.GetArrayFromImage(img_vol)
    raw_shape = img_array.shape

    # 阈值截取
    img_array = np.clip(img_array, ct_lower, ct_upper).astype(np.float32)

    # 裁剪掉零值区域
    region_array = np.zeros(img_array.shape)
    region_array[img_array > ct_lower] = 1
    labels, num = measure.label(region_array, return_num=True)
    regions = measure.regionprops(labels)
    regions_area = [regions[i].area for i in range(num)]
    region_num = regions_area.index(max(regions_area))
    bbox = regions[region_num].bbox
    img_array = img_array[bbox[0]:bbox[3]+1, bbox[1]:bbox[4]+1, bbox[2]:bbox[5]+1]

    # 归一化 -1 ~ 1
    img_array = 2 * (img_array - ct_lower) / (ct_upper - ct_lower) - 1.0

    # 重采样, lbl应使用最近邻插值
    img_spacing = img_vol.GetSpacing()
    img_array = ndimage.zoom(img_array, (img_spacing[2] / new_spacing[2],
                                         img_spacing[0] / new_spacing[0],
                                         img_spacing[1] / new_spacing[1]), order=3)

    # 保存数据
    new_img_array = img_array.astype(np.float32)

    new_img_vol = sitk.GetImageFromArray(new_img_array)
    new_img_vol.SetDirection(img_vol.GetDirection())
    new_img_vol.SetOrigin(img_vol.GetOrigin())
    new_img_vol.SetSpacing(tuple(new_spacing))

    new_img_name = os.path.join(sample_img_path, img_name)
    sitk.WriteImage(new_img_vol, new_img_name)

    ################################### label #####################################

    if lbl_vol is not None:
        lbl_array = sitk.GetArrayFromImage(lbl_vol)
        lbl_array = lbl_array[bbox[0]:bbox[3]+1, bbox[1]:bbox[4]+1, bbox[2]:bbox[5]+1]
        lbl_spacing = lbl_vol.GetSpacing()
        lbl_array = ndimage.zoom(lbl_array, (lbl_spacing[2] / new_spacing[2],
                                             lbl_spacing[0] / new_spacing[0],
                                             lbl_spacing[1] / new_spacing[1]), order=0)
        assert img_array.shape == lbl_array.shape

        new_lbl_array = lbl_array.astype(np.uint8)

        new_lbl_vol = sitk.GetImageFromArray(new_lbl_array)
        new_lbl_vol.SetDirection(lbl_vol.GetDirection())
        new_lbl_vol.SetOrigin(lbl_vol.GetOrigin())
        new_lbl_vol.SetSpacing(tuple(new_spacing))

        new_lbl_name = os.path.join(sample_lbl_path, img_name.replace('img', 'label'))

        sitk.WriteImage(new_lbl_vol, new_lbl_name)

    # 记录info
    if lbl_vol is not None:
        info.append([new_img_name, new_lbl_name, img_array.shape[0], img_array.shape[1], img_array.shape[2]])
    else:
        info.append([new_img_name, img_array.shape[0], img_array.shape[1], img_array.shape[2]])

    print('case:', img_name)
    print('raw space:', img_vol.GetSpacing(), 'new shape:', new_img_vol.GetSpacing())
    print('raw shape:', raw_shape, 'new shape:', img_array.shape)
    print('-------------------------')


if __name__ == "__main__":
    # raw training data path
    raw_path = r'D:\Projects\OrgansSegment\BTCV\RawData\Training'
    raw_img_path = os.path.join(raw_path, 'img')
    raw_lbl_path = os.path.join(raw_path, 'label')

    # training sample save path
    sample_path = r'D:\Projects\OrgansSegment\SegWithPatch\samples\Training'
    sample_img_path = os.path.join(sample_path, 'img')
    sample_lbl_path = os.path.join(sample_path, 'label')
    if os.path.exists(sample_path) is True:
        shutil.rmtree(sample_path)
    os.mkdir(sample_path)
    os.mkdir(sample_img_path)
    os.mkdir(sample_lbl_path)

    train_info_file = open('../../csv_files/btcv_train_info.csv', 'w', newline="")
    val_info_file = open('../../csv_files/btcv_val_info.csv', 'w', newline="")
    info = []

    for case in os.listdir(raw_img_path):
        # read image
        img_vol = sitk.ReadImage(os.path.join(raw_img_path, case))
        # read label
        lbl_case = case.replace('img', 'label')
        lbl_vol = sitk.ReadImage(os.path.join(raw_lbl_path, lbl_case))

        preprocess(case, img_vol, lbl_vol)

    # 分为训练集，验证集与测试集(24, 6)
    train_info_writer = csv.writer(train_info_file)
    val_info_writer = csv.writer(val_info_file)

    random.shuffle(info)
    for f in info[:6]:
        val_info_writer.writerow(f)
    for f in info[6:]:
        train_info_writer.writerow(f)

    ################################### label #####################################
    # raw testing data path
    raw_path = r'D:\Projects\OrgansSegment\BTCV\RawData\Testing'
    raw_img_path = os.path.join(raw_path, 'img')

    # test sample save path
    sample_path = r'D:\Projects\OrgansSegment\SegWithPatch\samples\Testing'
    sample_img_path = os.path.join(sample_path, 'img')
    if os.path.exists(sample_path) is True:
        shutil.rmtree(sample_path)
    os.mkdir(sample_path)
    os.mkdir(sample_img_path)

    test_info_file = open('../../csv_files/btcv_test_info.csv', 'w', newline="")
    info = []

    for case in os.listdir(raw_img_path):
        # read image
        img_vol = sitk.ReadImage(os.path.join(raw_img_path, case))

        preprocess(case, img_vol)

    test_info_writer = csv.writer(test_info_file)
    for f in info:
        test_info_writer.writerow(f)
