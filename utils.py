import torch
import numpy as np
from skimage import measure
import random
import os
import csv
import SimpleITK as sitk
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

project_root_path = os.path.abspath(os.path.dirname(__file__))
raw_path = "/home/hja/Projects/OrgansSegment/BTCA/RawData/Training"
preprocessed_save_path = os.path.join(project_root_path, "samples/BTCV/Training")
samples_info_file = os.path.join(project_root_path, "info_files/training_samples_info.csv")
dataset_info_file = os.path.join(project_root_path, "info_files/trainset_info.json")

num_steps_for_backward = 6  # how many training steps between two backward?
# number of patches in a batch = num_patches_volume * num_volumes_batch
num_patches_volume = 2
num_volumes_batch = 2
# validation batch size
val_batch_size = 8

padding_size = np.array([20, 40, 40])
patch_size = np.array([48, 160, 160])
crop_size = np.array([i * 3 / 2 for i in patch_size], dtype=np.int)

resume_training = False
module_dir = './module/td_unet80-0.668-0.601.pth'

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'
torch.backends.cudnn.benchmark = True
Epoch = 1000
iteration_every_epoch = 250
# 梯度累计，即每grad_accum_steps次iteration更新一次网络参数
grad_accum_steps = 2
inital_learning_rate = 1e-3
data_loader_processes = 4
# the weight for the batch from pseudo labels
batch_low_confidence_weight = 0.2

# 器官属性
organs_properties = {'organs_name': ['spleen', 'rkidny', 'lkidney', 'gallbladder', 'esophagus', 'liver', 'stomach',
                                     'aorta', 'vena', 'vein', 'pancreas', 'rgland', 'lgland'],
                     'organs_size': {1: 41254, 2: 21974, 3: 21790, 4: 3814, 5: 2182, 6: 236843, 7: 61189, 8: 13355,
                                     9: 11960, 10: 4672, 11: 11266, 12: 595, 13: 724},
                     'organs_weight': [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                     'num_organ': 13,
                     'sample_path': r'D:\Projects\OrgansSegment\SegWithPatch\samples\Training'}

organs_name = organs_properties['organs_name']
num_organ = organs_properties['num_organ']
organs_size = organs_properties['organs_size']
classes_name = ["bg"] + organs_name
class_weight = np.array([1] + organs_properties["organs_weight"])
class_weight = class_weight / np.max(class_weight)

network_configure = {'kernel_sizes': [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                     'features_channels': [32, 64, 128, 256, 320],
                     'down_strides': [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]}


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_weight_matrix(ps, sigma_scale=1. / 8):
    tmp = np.zeros(ps)
    center_coords = [i // 2 for i in ps]
    sigmas = [i * sigma_scale for i in ps]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


weight_matrix = calc_weight_matrix(patch_size)


def post_process(input):
    """
    分割结果的后处理：保留最大连通区域
    :param input: whole volume （D, W, H）
    :return: （D, W, H）
    """
    s = input.shape
    output = np.zeros((num_organ + 1, s[0], s[1], s[2]))
    for id in range(1, num_organ + 1):
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


def image_resize(old_image, new_shape, order, is_anisotropic):
    is_single_image = False
    if len(old_image.shape) == 3:  # (D, W, H)
        old_image = old_image[None]
        is_single_image = True

    old_type = old_image.dtype
    batch_size = old_image.shape[0]
    resized_image = np.zeros((batch_size, *new_shape), old_image.dtype)

    for b, img in enumerate(old_image):
        if not is_anisotropic:
            resized_image[b] = resize(img.astype(float), new_shape, order=order, preserve_range=True).astype(old_type)
        else:
            temp_image = np.zeros((img.shape[0], *new_shape[1:]))
            for i, slice in enumerate(img):
                temp_image[i] = resize(slice.astype(float), new_shape[1:], order=order, preserve_range=True)

            if img.shape[0] == new_shape[0]:
                resized_image[b] = temp_image.astype(old_type)
            else:
                scale = float(img.shape[0]) / new_shape[0]
                map_deps, map_rows, map_cols = np.mgrid[:new_shape[0], :new_shape[1], :new_shape[2]]
                map_deps = scale * (map_deps + 0.5) - 0.5
                coord_map = np.array([map_deps, map_rows, map_cols])
                resized_image[b] = map_coordinates(temp_image, coord_map, order=0, mode='nearest').astype(old_type)
    if is_single_image:
        return resized_image[0]
    else:
        return resized_image


# 根据training_samples_info.csv划分训练集与验证集
def split_train_val():
    val_amount = 6  # 验证集case数量
    csv_reader = csv.reader(open(samples_info_file, 'r'))
    all_samples_info = [row for row in csv_reader][1:]
    train_samples_info = []
    train_samples_name = []
    val_samples_info = []
    val_samples_name = []
    for info in all_samples_info:
        name = info[0].split("/")[-1]
        if int(name[5:7]) > 60:
            train_samples_info.append(info)
            train_samples_name.append(name)
        else:
            if len(val_samples_info) < 6:
                val_samples_info.append(info)
                val_samples_name.append(name)
            else:
                train_samples_info.append(info)
                train_samples_name.append(name)

    return train_samples_info, train_samples_name, val_samples_info, val_samples_name


def compute_steps_for_sliding_window(patch_size, image_size, step_size=0.5):
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


if __name__ == "__main__":
    # a = np.random.randint(1, 10, (2, 48, 256, 256)).astype(np.long)
    # resized_a = image_resize(a, (32, 200, 200), 0, is_anisotropic=True)
    # print(resized_a.shape, resized_a.dtype)

    # wm = calc_weight_matrix(patch_size)
    # print(patch_size)
    # print(wm.shape)
    # print(np.max(wm), np.min(wm))
    # center_coords = [i // 2 for i in patch_size]
    # print(center_coords)
    # print("center coord: ", wm[tuple(center_coords)])
    # print(wm[center_coords[0] - 5, center_coords[1] + 20, center_coords[2] - 30])
    # print(wm[center_coords[0] + 5, center_coords[1] - 20, center_coords[2] + 30])
    # print(wm[0, 0, 0])
    # print(wm[patch_size[0] - 1, patch_size[1] - 1, patch_size[2] - 1])
    # # print(wm)
    # train_samples_info, train_samples_name, _, _ = split_train_val()
    # print(train_samples_name)
    # for i in train_samples_info:
    #     print(i)
    #
    # _, _, val_samples_info, val_samples_name= split_train_val()
    # print(val_samples_name)
    # for i in val_samples_info:
    #     print(i)

    print(compute_steps_for_sliding_window(patch_size, (58, 221, 351)))
