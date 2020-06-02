from batchgenerators.dataloading import MultiThreadedAugmenter, SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms import Compose
import numpy as np
import csv
import json

import sys
sys.path.append("../")
from utils import *


class MyDataloader(SlimDataLoaderBase):
    current_counts = np.zeros(organs_properties["num_organ"])

    def __init__(self, class_weight):
        """
        data loader
        :param is_train: dataloader for training set or validation set
        :param folder: 0 - 4 (5-folder cross validation)
        :param patch_size: a tuple
        :param batch_size: int
        """
        super(MyDataloader, self).__init__(None, num_patches_volume * num_volumes_batch, None)
        # assert 0 <= folder <= 4, "only support 5-folder cross validation"

        # load infos of data from training_samples_info.csv
        csv_reader = csv.reader(open(samples_info_file, 'r'))
        all_samples_info = [row for row in csv_reader][1:]
        self._data = all_samples_info
        # samples_every_folder = len(all_samples_info) // 5
        # if not is_train:
        #     self._data = all_samples_info[folder * samples_every_folder: (folder + 1) * samples_every_folder]
        # else:
        #     self._data = all_samples_info[:folder * samples_every_folder] + \
        #                  all_samples_info[(folder + 1) * samples_every_folder:]

        self.class_weight = class_weight
        self.num_class = len(class_weight)
        self.current_position = 0
        self.was_initialized = False
        self.stop_iteration = 0

    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True
        self.stop_iteration = iteration_every_epoch - (data_loader_processes - self.thread_id)
        print("data loader thread id: %d, will stop before %d th iteration." % (self.thread_id, self.stop_iteration))

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        self.current_position = self.current_position + self.number_of_threads_in_multithreaded
        if self.current_position > self.stop_iteration:
            raise StopIteration

        if np.sum(MyDataloader.current_counts) == 0:
            current_weight = np.zeros(len(self.class_weight))
        else:
            current_weight = MyDataloader.current_counts / np.sum(MyDataloader.current_counts)

        data_patches = []
        seg_patches = []
        image_names = []
        batch_class_ids = []
        for i in range(num_volumes_batch):
            selected_ids = np.random.choice(range(len(self._data)), 1)[0]
            selected_samples = self._data[selected_ids]

            image = np.load(selected_samples[0])
            segmentation = np.load(selected_samples[1])
            for j in range(num_patches_volume):
                # select a patch in which the class with the highest weight is contained
                class_id = np.argmax(self.class_weight - current_weight) + 1
                data_patch, seg_patch, contained_ids = self.crop_patch(image, segmentation, class_id)
                data_patches.append(data_patch[None])
                seg_patches.append(seg_patch[None])
                image_names.append(selected_samples[0].split("/")[-1])
                
                batch_class_ids += contained_ids
                for id in contained_ids:
                    MyDataloader.current_counts[id - 1] += 1
        data_patches = np.array(data_patches).astype(np.float32)
        seg_patches = np.array(seg_patches).astype(np.float32)
        return {'data': data_patches, 'seg': seg_patches, 'image_names': image_names, 'batch_class_ids': batch_class_ids}

    def crop_patch(self, image, label, class_id):
        """
        crop patch from image and label, the centre of the patch should be the class_id organ
        :param self:
        :param image:
        :param label:
        :param class_id: the class with the highest weight
        :return: image patch, label patch, the ids of organs witch covering at least 10% pixels in the sampled patch
        """
        shape = image.shape
        assert image.shape == label.shape
        # find a point as the centre of patch
        class_points = np.argwhere(label == class_id)
        # class_points = list(filter(lambda point:
        #                       np.all([crop_size[0]//2 <= point[0] <= s[0] - crop_size[0]//2,
        #                               crop_size[1]//2 <= point[1] <= s[1] - crop_size[1]//2,
        #                               crop_size[2]//2 <= point[2] <= s[2] - crop_size[2]//2]), class_points))
        if len(class_points) == 0:
            class_id = np.random.choice(label[label != 0])
            class_points = np.argwhere(label == class_id)

        centre_point = class_points[np.random.choice(range(len(class_points)))]
        # shift to avoid crossing border
        boder_margin = [i//2 for i in patch_size]
        centre_point = [centre_point[i] if centre_point[i] <= s - boder_margin[i] else s - boder_margin[i] for i, s in enumerate(shape)]
        centre_point = [centre_point[i] if centre_point[i] >= boder_margin[i] else boder_margin[i] for i, s in enumerate(shape)]

        slice_z = slice(centre_point[0] - boder_margin[0], centre_point[0] + boder_margin[0])
        slice_x = slice(centre_point[1] - boder_margin[1], centre_point[1] + boder_margin[1])
        slice_y = slice(centre_point[2] - boder_margin[2], centre_point[2] + boder_margin[2])
        data_patch = image[slice_z, slice_x, slice_y]
        seg_patch = label[slice_z, slice_x, slice_y]

        contained_class_id = [class_id]
        num_pixel_thresh = 0.1 * np.product(patch_size)
        for cid in range(1, self.num_class+1):
            if np.sum(seg_patch == cid) > num_pixel_thresh:
                contained_class_id.append(cid)

        return data_patch, seg_patch, contained_class_id


def get_train_transform():
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
#     tr_transforms.append(
#         SpatialTransform_2(
#             patch_size, 0,
#             do_elastic_deform=False, deformation_scale=(0.25, 0.5),
#             do_rotation=False,
#             angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
#             angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
#             angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
#             do_scale=False, scale=(0.75, 1.25),
#             border_mode_data='constant', border_cval_data=0,
#             border_mode_seg='constant', border_cval_seg=0,
#             order_seg=1, order_data=3,
#             random_crop=False,
#             p_el_per_sample=0, p_rot_per_sample=0.1, p_scale_per_sample=0.1
#         )
#     )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    
    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=True, per_channel=True, p_per_sample=0.15))
    
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                     p_per_channel=0.5,
                                                     order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                     ignore_axes=(0,)))
    
    tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3), p_per_sample=0.15))
    
    tr_transforms.append(GaussianBlurTransform((0.5, 1.5), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15))


    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_data_loader(class_weights, num_processes):
    dl = MyDataloader(class_weights)
    trans = get_train_transform()

    return MultiThreadedAugmenter(dl, trans, num_processes)


if __name__ == "__main__":
    ml = get_data_loader(class_weight, 3)

    batches = []
    dataset_info = json.load(open(dataset_info_file, 'r'))
    save_id = 0
    median_spacing = dataset_info['median_spacing']
    for i, batch in enumerate(ml):
        print(i, batch['data'].shape, batch['seg'].shape, batch['image_names'], batch['batch_class_ids'])
        bs = batch['data'].shape[0]
        batches.append(batch['seg'])

        for j in range(bs):
            img_vol = sitk.GetImageFromArray(batch['data'][j].squeeze())
            lbl_vol = sitk.GetImageFromArray(batch['seg'][j].squeeze())
            img_vol.SetSpacing(median_spacing)
            lbl_vol.SetSpacing(median_spacing)
            sitk.WriteImage(img_vol, "img_%s.nii.gz" % save_id)
            sitk.WriteImage(lbl_vol, "lbl_%s.nii.gz" % save_id)
            save_id += 1

        if i == 239:
            break
        # current_counts[batch] += 1
        # dl.set_current_counts(current_counts)
        # print(dl.get_cur())


