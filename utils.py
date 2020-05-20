import torch
import numpy as np
from skimage import measure
import random
import os
import SimpleITK as sitk


project_root_path = os.path.abspath(os.path.dirname(__file__))
preprocessed_save_path = os.path.join(project_root_path, "samples/BTCV/Training")
samples_info_file = os.path.join(project_root_path, "info_files/training_samples_info.csv")
dataset_info_file = os.path.join(project_root_path, "info_files/trainset_info.json")

num_steps_for_backward = 6  # how many training steps between two backward?
# number of patches in a batch = num_patches_volume * num_volumes_batch
num_patches_volume = 2
num_volumes_batch = 2

patch_size = np.array([48, 128, 128])
crop_size = np.array([i * 3 / 2 for i in patch_size], dtype=np.int)

# 器官属性
organs_properties = {'organs_name': ['spleen', 'rkidny', 'lkidney', 'gallbladder', 'esophagus', 'liver', 'stomach',
                                     'aorta', 'vena', 'vein', 'pancreas', 'rgland', 'lgland'],
                     'organs_size': {1: 41254, 2: 21974, 3: 21790, 4: 3814, 5: 2182, 6: 236843, 7: 61189, 8: 13355,
                                     9: 11960, 10: 4672, 11: 11266, 12: 595, 13: 724},
                     'organs_weight': [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                     'num_organ': 13,
                     'sample_path': r'D:\Projects\OrgansSegment\SegWithPatch\samples\Training'}
# organs_properties = {'organs_index': [1, 3, 4, 5, 6, 7, 11, 14],
#                      'organs_name': ['spleen', 'left kidney', 'gallbladder', 'esophagus',
#                                      'liver', 'stomach', 'pancreas', 'duodenum'],
#                      'organs_size': {1: 33969.37777777778, 3: 21083.43820224719, 4: 3348.8214285714284,
#                                      5: 1916.685393258427, 6: 208806.8777777778, 7: 50836.01111111111,
#                                      11: 9410.111111111111, 14: 11118.544444444444},
#                      'num_organ': 8}

class_weight = np.array(organs_properties["organs_weight"])
class_weight = class_weight / np.sum(class_weight)
current_counts = np.zeros(len(class_weight))

organs_name = organs_properties['organs_name']
num_organ = organs_properties['num_organ']
organs_size = organs_properties['organs_size']


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_weight_matrix(sample_size):
    median = sample_size / 2 - 0.5

    distance_matrix = np.zeros((sample_size, sample_size, sample_size), dtype=np.float)
    for i in range(sample_size):
        for j in range(sample_size):
            for k in range(sample_size):
                distance_matrix[i, j, k] = int(abs(i - median)) + \
                                           int(abs(j - median)) + \
                                           int(abs(k - median)) + 1

    pair_sum = np.max(distance_matrix) + np.min(distance_matrix)  # sum of distance of every pair
    total_sum = pair_sum * 4  # 4 pairs
    weight_matrix = (pair_sum - distance_matrix) / total_sum
    return weight_matrix


def post_process(input):
    """
    分割结果的后处理：保留最大连通区域
    :param input: whole volume （D, W, H）
    :return: （D, W, H）
    """
    s = input.shape
    output = np.zeros((num_organ+1, s[0], s[1], s[2]))
    for id in range(1, num_organ+1):
        org_seg = (input == id) + .0
        if (org_seg == .0).all():
            continue
        labels, num = measure.label(org_seg, return_num=True)
        regions = measure.regionprops(labels)
        regions_area = [regions[i].area for i in range(num)]

        # omit_region_id = []
        # for rid, area in enumerate(regions_area):
        #     if area < 0.1 * organs_size[organs_index[id - 1]]:  # 记录区域面积小于10%*器官尺寸的区域的id
        #         omit_region_id.append(rid)
        # for idx in omit_region_id:
        #     org_seg[labels == (idx+1)] = 0

        region_num = regions_area.index(max(regions_area)) + 1  # 记录面积最大的区域，不会计算background(0)
        org_seg[labels == region_num] = 1
        org_seg[labels != region_num] = 0

        output[id, :, :, :] = org_seg

    return np.argmax(output, axis=0)


def read_dicom(path):
    # read dicom series
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    # image_array = sitk.GetArrayFromImage(image)
    # spacing = image.GetSpacing()
    # origin = image.GetOrigin()
    # dims = image.GetDimension()

    return image


def read_nii(path, type=None):
    # read nifit
    if type == None:
        return sitk.ReadImage(path)
    else:
        return sitk.ReadImage(path, type)


if __name__ == "__main__":
    a = torch.randint(1, 10, (6, 9, 48, 256, 256))
    b = torch.randint(1, 10, (6, 48, 256, 256))

