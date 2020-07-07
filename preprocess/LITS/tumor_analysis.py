import os
import SimpleITK as sitk
import numpy as np
from collections import Counter
from skimage.transform import resize
from skimage.feature import greycoprops, greycomatrix
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt


def calc_greycomatrix_features(data, label, min_gray, max_gray):
    data = np.clip(data, min_gray, max_gray)
    points = np.argwhere(label == 2)
    points = [tuple(p) for p in points]
    assert len(points) > 2
    gray_level = max_gray - min_gray + 1
    offset = [[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, -1, 1], [1, 1, 0], [1, 0, 0], [1, -1, 0]]
    grey_comatrix = np.zeros((gray_level, gray_level, 1, len(offset))).astype(np.uint32)

    # compute grey matrix
    for p in points:
        for k, o in enumerate(offset):
            p1 = tuple(i + j for i, j in zip(p, o))
            if p1 in points:
                grey_comatrix[data[p] - min_gray, data[p1] - min_gray, 0, k] += 1

    # compute features based on grey matrix
    features = {
        'contrast': np.mean(greycoprops(grey_comatrix, 'contrast')),
        'dissimilarity': np.mean(greycoprops(grey_comatrix, 'dissimilarity')),
        'homogeneity': np.mean(greycoprops(grey_comatrix, 'homogeneity')),
        'ASM': np.mean(greycoprops(grey_comatrix, 'ASM')),
        'energy': np.mean(greycoprops(grey_comatrix, 'energy')),
        'correlation': np.mean(greycoprops(grey_comatrix, 'correlation')),
    }

    return features


def intensity_statistic(intensity_dict=None):
    """
    calculate the clip intensity and statistic information(mean and variance)
    :param intensity_dict:
    :return: clip min intensity, clip max intensity, mean and variance
    """
    print("\nCalculate the intensity threshold, mean and variance...")
    # sort by keys(intensity)
    intensity_items = sorted(intensity_dict.items())
    num_pixels = sum(intensity_dict.values())

    # get clip intensity (0.5% and 99.5%)
    clip_min_counts = 0.005 * num_pixels
    clip_max_counts = num_pixels - 0.995 * num_pixels

    temp_count = 0
    min_index = 0
    for min_index, items in enumerate(intensity_items):
        temp_count += intensity_items[min_index][1]
        if temp_count >= clip_min_counts:
            break
    clip_min_intensity = intensity_items[min_index][0]
    for i in range(min_index):
        intensity_items.pop(0)

    temp_count = 0
    max_index = 0
    intensity_items = intensity_items[::-1]
    for max_index, items in enumerate(intensity_items):
        temp_count += intensity_items[max_index][1]
        if temp_count >= clip_max_counts:
            break
    clip_max_intensity = intensity_items[max_index][0]
    for i in range(max_index):
        intensity_items.pop(0)

    # calculate mean and variance
    num_pixels = sum([x[1] for x in intensity_items])
    mean = sum([(x[0] / 1000) * x[1] for x in intensity_items]) / num_pixels * 1000
    variance = sum(x[1] * (x[0] - mean) ** 2 for x in intensity_items) / num_pixels
    print("clip_min_intensity: %d, clip_max_intensity: %d, mean: %.3f, variance: %.3f"
          % (int(clip_min_intensity), int(clip_max_intensity), mean, variance))

    return dict(intensity_items), int(clip_min_intensity), int(clip_max_intensity), mean, variance


def image_resize(old_image, old_spacing, new_spacing, is_seg):
    old_shape = old_image.shape
    new_shape = np.round((np.array([old_spacing[2] / new_spacing[2],
                                    old_spacing[0] / new_spacing[0],
                                    old_spacing[1] / new_spacing[1]]).astype(float) * old_shape)).astype(int)

    if np.max(old_spacing) / np.min(old_spacing) < 3:  # 各向同性
        resized_image = resize(old_image.astype(float), new_shape, order=3, preserve_range=True)
    else:
        assert np.argmax(old_spacing) == 2  # 各向异性，z轴spacing最大
        temp_image = np.zeros((old_image.shape[0], *new_shape[1:]))
        for i, slice in enumerate(old_image):
            temp_image[i] = resize(slice.astype(float), new_shape[1:], order=3, preserve_range=True)

        if old_image.shape[0] == new_shape[0]:
            resized_image = temp_image
        else:
            scale = float(old_image.shape[0]) / new_shape[0]
            map_deps, map_rows, map_cols = np.mgrid[:new_shape[0], :new_shape[1], :new_shape[2]]
            map_deps = scale * (map_deps + 0.5) - 0.5
            coord_map = np.array([map_deps, map_rows, map_cols])
            resized_image = map_coordinates(temp_image, coord_map, order=3)

    if is_seg:
        return (resized_image + 0.5).astype(np.uint8)
    else:
        return (resized_image + 0.5).astype(np.int)


if __name__ == "__main__":
    tumor_root = r"/home/space/tmp_sjr/tumor_patch"
    tumors_name = list(filter(lambda x: x[:6] == 'volume', os.listdir(tumor_root)))
    segments_name = list(filter(lambda x: x[:12] == 'segmentation', os.listdir(tumor_root)))
    target_spacing = [1.0, 1.0, 1.0]

    # 尺寸特征
    sizes = []
    # 形状特征
    bb_ratios = []
    curvatures_x = Counter({})
    curvatures_y = Counter({})
    curvatures_z = Counter({})
    # 灰度特征
    intensities = []
    means = []
    variances = []
    # 纹理特征
    contrasts = Counter({})

    for tumor in tumors_name:
        tumor_vol = sitk.ReadImage(os.path.join(tumor_root, tumor))
        segment_vol = sitk.ReadImage(os.path.join(tumor_root, tumor.replace("volume", "segmentation")))
        segment_array = sitk.GetArrayFromImage(segment_vol)

        tumor_array = image_resize(sitk.GetArrayFromImage(tumor_vol), tumor_vol.GetSpacing(), target_spacing,
                                   is_seg=False)
        segment_array = image_resize(segment_array, segment_vol.GetSpacing(), target_spacing, is_seg=True)

        size = np.sum(segment_array != 0)
        if size == 0:
            continue
        bg_range = np.where(segment_array != 0)
        z_min = np.min(bg_range[0])
        z_max = np.max(bg_range[0]) + 1
        x_min = np.min(bg_range[1])
        x_max = np.max(bg_range[1]) + 1
        y_min = np.min(bg_range[2])
        y_max = np.max(bg_range[2]) + 1
        bb_ratio = size / ((z_max - z_min) * (x_max - x_min) * (y_max - y_min))
        assert bb_ratio >= 1
        bb_ratios.append(bb_ratio)

        # sizes.append(size)
        np.round()

    print(len(bb_ratios))
    print(np.min(bb_ratios))
    print(np.max(bb_ratios))
    plt.hist(bb_ratios, bins=50, density=1)
    plt.title()
    plt.show()
    plt.bar()

    greycomatrix()
