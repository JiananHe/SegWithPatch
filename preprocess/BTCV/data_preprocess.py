from collections import Counter
import csv
import json
import shutil
from skimage.transform import resize
from batchgenerators.utilities.file_and_folder_operations import *

import sys 
sys.path.append("../../") 
from utils import *


def check_path():
    """
    check the paths of image and label in preprocessed_save_path
    """
    paths_for_check = [os.path.join(preprocessed_save_path, "img"), os.path.join(preprocessed_save_path, "label")]
    for path in paths_for_check:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)


def get_median_spacing(images_path, labels_path, format="nii"):
    """
    statistic median spacing of images in train set
    :param images_path: path of raw images
    :param labels_path: path of labels
    :param format: nii or dcm
    :return: median_spacing in z, x, y
    """
    print("Calculate the median spacing for raw images...")
    images_names = os.listdir(images_path)
    spacings = []
    for image_name in images_names:
        # read volumes of images and labels
        if format == "nii":
            image_vol = read_nii(os.path.join(images_path, image_name))
            label_vol = read_nii(os.path.join(labels_path, match_lbl_name(image_name)))
        elif format == "dcm":
            image_vol = read_dicom(os.path.join(images_path, image_name))
            label_vol = read_dicom(os.path.join(labels_path, match_lbl_name(image_name)))
        else:
            raise NameError("unsupported format, only support dicom and nifit") from Exception

        # record the spacing
        image_spacing = [round(i, 3) for i in image_vol.GetSpacing()]
        label_spacing = [round(i, 3) for i in label_vol.GetSpacing()]
        assert image_spacing == label_spacing, "the spacing of image and label in %s are different" % image_name
        spacings.append(list(image_spacing))

    spacings = np.array(spacings)
    median_spacing = [np.median(spacings[:, 0]), np.median(spacings[:, 1]), np.median(spacings[:, 2])]
    print("the median spacing is: ", median_spacing)
    return median_spacing


def save_volume(save_array, volume_properties, save_path):
    """
    save an array as a volume with the properties
    :param save_array: the array to be saved
    :param volume_properties: spacing, direction and origin
    :param save_path: the path for saving
    :return: None
    """
    save_volume = sitk.GetImageFromArray(save_array)
    save_volume.SetSpacing(volume_properties[0])
    save_volume.SetDirection(volume_properties[1])
    save_volume.SetOrigin(volume_properties[2])
    sitk.WriteImage(save_volume, save_path)


def crop_roi(image, label):
    """
    crop the image and label to the roi
    :param image: volume of CT image
    :param label: volume of label
    :return: cropped image and label, cropping coordinates
    """
    assert image.shape == label.shape
    s = image.shape

    z_min = np.min(np.where(label != 0)[0])
    z_min = z_min - padding_size[0] // 2 if z_min - padding_size[0] // 2 > 0 else 0
    z_max = np.max(np.where(label != 0)[0]) + 1
    z_max = z_max + padding_size[0] // 2 if z_max + padding_size[0] // 2 < s[0] else s[0]

    x_min = np.min(np.where(label != 0)[1])
    x_min = x_min - padding_size[1] // 2 if x_min - padding_size[1] // 2 > 0 else 0
    x_max = np.max(np.where(label != 0)[1]) + 1
    x_max = x_max + padding_size[1] // 2 if x_max + padding_size[1] // 2 < s[1] else s[1]

    y_min = np.min(np.where(label != 0)[2])
    y_min = y_min - padding_size[2] // 2 if y_min - padding_size[2] // 2 > 0 else 0
    y_max = np.max(np.where(label != 0)[2]) + 1
    y_max = y_max + padding_size[2] // 2 if y_max + padding_size[2] // 2 < s[2] else s[2]

    image_cropped = image[z_min:z_max, x_min:x_max, y_min:y_max]
    label_cropped = label[z_min:z_max, x_min:x_max, y_min:y_max]
    return image_cropped, label_cropped, [z_min, z_max, x_min, x_max, y_min, y_max]


def register_dataset(fixed_image_name):
    """
    register the fixed image (the broadest image) with all other preprocessed images
    :param fixed_image_name:
    :return: None (save the deformation filed)
    """
    fixed_image = read_nii(os.path.join(preprocessed_save_path, "img", fixed_image_name), sitk.sitkFloat32)
    # fixed_image_mask = read_nii(os.path.join(preprocessed_save_path, "label",
    #                                          match_lbl_name(fixed_image_name)))
    # fixed_image_mask = fixed_image_mask != 0

    registration_method = sitk.ImageRegistrationMethod()

    # registration_method.SetMetricAsDemons(10)  # intensities are equal if the difference is less than 10HU
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # registration_method.SetMetricFixedMask(fixed_image_mask)

    for moving_image_name in os.listdir(os.path.join(preprocessed_save_path, "img")):
        if moving_image_name == fixed_image_name:
            continue
        moving_image = read_nii(os.path.join(preprocessed_save_path, "img", moving_image_name), sitk.sitkFloat32)

        # Affine transform
        # initial_transform = sitk.AffineTransform(3)
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.AffineTransform(3),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Create initial identity transformation.
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
        # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
        optimize_transform = sitk.DisplacementFieldTransform(
            transform_to_displacment_field_filter.Execute(sitk.Transform()))

        # Regularization (update field - viscous, total field - elastic).
        optimize_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

        # Set the initial moving and optimized transforms.
        registration_method.SetMovingInitialTransform(initial_transform)
        registration_method.SetInitialTransform(optimize_transform)

        registration_method.Execute(fixed_image, moving_image)

        final_transform = sitk.Transform(optimize_transform)
        final_transform.AddTransform(initial_transform)
        print(final_transform)

        out_movingImage = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear,
                                         0.0, moving_image.GetPixelIDValue())

        sitk.WriteImage(out_movingImage, os.path.join(preprocessed_save_path, "registration", moving_image_name))


# def resample_data_or_seg(data, orig_spacing, new_spacing, is_seg, anisotropy_threshold):
#     """
#     resample an image or a segmentation.
#     if the maximum spacing / minimum spacing > 3, then resampling in-plane slice along other two smaller axises with
#     3-order spline interpolation, and resampling with the nearest interpolation along the axis with the maximum spacing.
#     Otherwise resmapling the whole 3D data with 3-order spline interpolation (image) or the nearest interpolation (segmentation).
#     :param data: 3D numpy array (an image or a segmentation)
#     :param orig_spacing: original spacing
#     :param new_spacing: new spacing
#     :param is_seg: bool
#     :param anisotropy_threshold: 3
#     :return: resampled data
#     """
#     assert len(data.shape) == 3, "data must be (z, x, y)"
#     # resample function for image or segmentation
#     if is_seg:
#         resize_fn = resize_segmentation
#         order = 0
#         cval = -1
#         kwargs = OrderedDict()
#     else:
#         resize_fn = resize
#         order = 3
#         cval = 0
#         kwargs = {'mode': 'edge', 'anti_aliasing': False}
#
#     assert np.where(max(new_spacing) / np.array(new_spacing) == 1)[0] == 0  # Note: z spacing should be the largest
#     is_anisotropy = (np.max(orig_spacing) / np.min(orig_spacing)) > 3 or (np.max(new_spacing) / np.min(new_spacing)) > anisotropy_threshold
#
#     dtype_data = data.dtype
#     data = data.astype(float)
#     shape = np.array(data.shape)
#     new_shape = np.round(((np.array(orig_spacing) / np.array(new_spacing)).astype(float) * shape)).astype(int)
#
#     if np.any(shape != new_shape):
#         if is_anisotropy:
#             reshaped_data = []
#             for slice_id in range(shape[0]):
#                 reshaped_data.append(resize_fn(data[slice_id], new_shape[1:], order, cval=cval, **kwargs))
#
#             reshaped_data = np.stack(reshaped_data)
#             assert new_shape[1:] == reshaped_data.shape[1:]
#             if shape[0] != new_shape[0]:
#                 dims, rows, cols = new_shape
#                 orig_dims = reshaped_data.shape[0]
#
#                 dim_scale = float(orig_dims) / dims
#
#                 map_dims, map_rows, map_cols = np.mgrid[:dims, :rows, :cols]  # np.mgrid: return shape (3, dims, rows, cols)
#                 map_dims = dim_scale * (map_dims + 0.5) - 0.5
#
#                 coord_map = np.array([map_rows, map_cols, map_dims])
#                 if not is_seg:
#                     reshaped_final_data = map_coordinates(reshaped_data, coord_map, order=0, cval=cval, mode='nearest')
#                 else:
#                     unique_labels = np.unique(reshaped_data)
#                     reshaped = np.zeros(new_shape, dtype=dtype_data)
#
#                     for i, cl in enumerate(unique_labels):
#                         reshaped_multihot = np.round(
#                             map_coordinates((reshaped_data == cl).astype(float), coord_map, order=0, cval=cval, mode='nearest'))
#                         reshaped[reshaped_multihot > 0.5] = cl
#                     reshaped_final_data = reshaped
#             else:
#                 reshaped_final_data.append(reshaped_data[None])
#         reshaped_final_data = np.vstack(reshaped_final_data)
#         else:
#             print("no separate z, order", order)
#             reshaped = []
#             for c in range(data.shape[0]):
#                 reshaped.append(resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None])
#             reshaped_final_data = np.vstack(reshaped)
#         return reshaped_final_data.astype(dtype_data)
#     else:
#         print("no resampling necessary")
#         return data


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

    return int(clip_min_intensity), int(clip_max_intensity), mean, variance


def intensity_clip_norm(clip_min_intensity, clip_max_intensity, mean, std_variance):
    """
    clip and normalize the intensity of all samples according to the data set info
    :param clip_min_intensity: the low intensity threshold
    :param clip_max_intensity: the high intensity threshold
    :param mean: mean
    :param std_variance: standard variance
    :return: None
    """
    print("\nClip and normalize intensity...")
    for image_name in os.listdir(os.path.join(preprocessed_save_path, "img")):
        print(image_name)
        image_vol = read_nii(os.path.join(preprocessed_save_path, "img", image_name))
        image_array = sitk.GetArrayFromImage(image_vol)
        image_array = np.clip(image_array, clip_min_intensity, clip_max_intensity)
        image_array = (image_array - mean) / std_variance

        save_volume(image_array, [image_vol.GetSpacing(), image_vol.GetDirection(), image_vol.GetOrigin()],
                    os.path.join(preprocessed_save_path, "img", image_name))
        np.save(os.path.join(preprocessed_save_path, "img", image_name).replace("nii.gz", "npy"), image_array)


def train_dataset_preprocess(images_path, labels_path, format='nii'):
    """
    preprocess images and labels for training dataset Statistical grayscale information
    :param images_path: path of images
    :param labels_path: path of labels
    :param format: nii or dcm
    :return: None
    """
    if os.path.exists(dataset_info_file):
        dataset_info = json.load(open(dataset_info_file, "r"))
        print(dataset_info)
        return
    else:
        check_path()

    # check paths and names of labels and images
    check_path()
    images_names = os.listdir(images_path)
    assert os.listdir(labels_path) == [match_lbl_name(img_name) for img_name in images_names]

    # calculate the median spacing and record the broadest sample and intensities
    dataset_info = {}
    # median_spacing = get_median_spacing(images_path, labels_path)
    median_spacing = [0.76, 0.76, 3.0]
    # broadest_sample_shape = 0
    # broadest_sample = ""
    # intensities_counter = Counter({})

    # samples infos to be recorded
    samples_infos = []
    samples_info_writer = csv.writer(open(samples_info_file, "w", newline=""))
    samples_info_writer.writerow(["processed_image", "processed_label", "raw_spacing", "raw_shape", "cropped_shape", "cropped_coordinates"])

    print("\nPreProcess training set...")
    for image_name in images_names:
        # read volumes of images and labels
        if format == "nii":
            image_vol = read_nii(os.path.join(images_path, image_name))
            label_vol = read_nii(os.path.join(labels_path, match_lbl_name(image_name)))
        elif format == "dcm":
            image_vol = read_dicom(os.path.join(images_path, image_name))
            label_vol = read_dicom(os.path.join(labels_path, match_lbl_name(image_name)))
        else:
            raise NameError("unsupported format, only support dicom and nifit") from Exception

        # get numpy array
        image_array = sitk.GetArrayFromImage(image_vol)
        label_array = sitk.GetArrayFromImage(label_vol)
        image_dtype = image_array.dtype
        image_array = image_array.astype(float)
        label_dtype = label_array.dtype
        label_array = label_array.astype(float)
        print(image_name, image_dtype)
        assert image_array.shape == label_array.shape, "the shapes of image and label in %s are different" % image_name

        # step 1: crop to roi
        image_cropped, label_cropped, crop_coord = crop_roi(image_array, label_array)

        # step 2: resample to median spacing
        raw_image_spacing = image_vol.GetSpacing()
        image_resampled = image_resample(image_cropped, raw_image_spacing, median_spacing, 3, True)
        label_resampled = image_resample(label_cropped, raw_image_spacing, median_spacing, 0, True)

        assert image_resampled.shape == label_resampled.shape
        image_resampled = image_resampled.astype(image_dtype)
        label_resampled = label_resampled.astype(label_dtype)
        print("new shape:", image_resampled.shape)

        # # statistic max shape
        # if broadest_sample_shape < np.product(image_cropped.shape):
        #     broadest_sample_shape = np.product(image_cropped.shape)
        #     broadest_sample = image_name
        # # statistic gray info
        # intensities_counter += Counter(image_cropped[label_cropped != 0])

        # save the cropped and resampled data temporarily
        image_save_path = os.path.abspath(os.path.join(preprocessed_save_path, "img", image_name))
        label_save_path = os.path.abspath(os.path.join(preprocessed_save_path, "label", match_lbl_name(image_name).replace("img", "label")))

        save_volume(image_resampled, [median_spacing, image_vol.GetDirection(), image_vol.GetOrigin()], image_save_path)
        save_volume(label_resampled, [median_spacing, label_vol.GetDirection(), label_vol.GetOrigin()], label_save_path)
        np.save(label_save_path.replace("nii.gz", "npy"), label_resampled)

        samples_infos.append([image_save_path.replace("nii.gz", "npy"), label_save_path.replace("nii.gz", "npy"),
                              raw_image_spacing, image_array.shape, image_cropped.shape, crop_coord])

    # record the information about save path and cropping coordinates in csv file
    # np.random.shuffle(samples_infos)
    for info in samples_infos:
        samples_info_writer.writerow(info)

    # register
    # register_dataset(broadest_sample)

    # calculate the clip intensity and statistic information(mean and variance)
    # clip_min_intensity, clip_max_intensity, mean, variance = intensity_statistic(dict(intensities_counter))
    clip_min_intensity = -958
    clip_max_intensity = 327
    mean = 82.92
    std_variance = 136.97

    # step 3: clip and normalize intensity
    intensity_clip_norm(clip_min_intensity, clip_max_intensity, mean, std_variance)

    # record data set infomation
    dataset_info["median_spacing"] = median_spacing
    # dataset_info["broadest_sample"] = broadest_sample
    dataset_info["clip_min_intensity"] = clip_min_intensity
    dataset_info["clip_max_intensity"] = clip_max_intensity
    dataset_info["mean"] = mean
    dataset_info["std_variance"] = std_variance
    with open(dataset_info_file, "w") as f:
        json.dump(dataset_info, f)


if __name__ == "__main__":
    raw_img_path = os.path.join(raw_path, "img")
    raw_lbl_path = os.path.join(raw_path, "label")
    print(raw_lbl_path)
    train_dataset_preprocess(raw_img_path, raw_lbl_path)
    # dataset_info = json.load(open(dataset_info_file, "r"))
    # intensity_clip_norm(dataset_info["clip_min_intensity"], dataset_info["clip_max_intensity"], dataset_info["mean"], dataset_info["variance"])
