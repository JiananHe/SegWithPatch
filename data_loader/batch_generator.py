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
    # current_counts = [0] * (organs_properties["num_organ"] + 1)

    def __init__(self, num_threads_in_mt, class_weight=None):
        super(MyDataloader, self).__init__(None, num_patches_volume * num_volumes_batch, num_threads_in_mt)
        # assert 0 <= folder <= 4, "only support 5-folder cross validation"

        # load infos of data from training_samples_info.csv
        train_samples_info, train_samples_name, _, _ = split_train_val()
        print("samples for training: ", train_samples_name)
        self._data = train_samples_info

        self.class_weight = class_weight
        self.num_class = num_organ + 1
        self.generator_patch_size = get_generator_patch_size(patch_size, rotation_x, rotation_y, rotation_z, range_scale)
        self.current_position = 0
        self.was_initialized = False

        # set the range of weights for sampling, ignore background
        # self.organ_weight = class_weight[1:]
        # self.organ_weight /= np.sum(self.organ_weight)
        # self.organ_weight_range = np.zeros(self.num_class)
        # for i in range(1, len(self.organ_weight)+1):
        #     self.organ_weight_range[i] = self.organ_weight_range[i-1]+self.organ_weight[i-1]

    def reset(self):
        # if self.was_initialized:
        #     print("total counts for every class: ", MyDataloader.current_counts)
        self.current_position = self.thread_id
        self.was_initialized = True

    # def calc_sample_class_id(self):
    #     current_random = np.random.rand()
    #     for r in range(len(self.organ_weight)):
    #         if current_random >= self.organ_weight_range[r] and current_random < self.organ_weight_range[r+1]:
    #             return r + 1
        
        # class_weight_sorted = sorted(self.class_weight)
        # class_weight_order = [class_weight_sorted.index(i) for i in self.class_weight]
        # count_sorted = sorted(MyDataloader.current_counts)
        # count_order = [count_sorted.index(i) for i in MyDataloader.current_counts]

        # diff_order = np.subtract(class_weight_order, count_order)
        # under_sampled_id = np.argwhere(np.array(diff_order) > 0).squeeze()
        # return np.random.choice(under_sampled_id)

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        self.current_position += self.number_of_threads_in_multithreaded
        if self.current_position > iteration_every_epoch + (self.number_of_threads_in_multithreaded - 1):
            self.reset()
            raise StopIteration

        data_patches = []
        seg_patches = []
        image_names = []
        # batch_class_ids = []
        for i in range(num_volumes_batch):
            selected_ids = np.random.choice(range(len(self._data)), 1)[0]
            selected_samples = self._data[selected_ids]

            image = np.load(selected_samples[0]).astype(np.float)
            segmentation = np.load(selected_samples[1]).astype(np.int)
            for j in range(num_patches_volume):
                # select a patch in which the class with the highest weight is contained
                # class_id = np.argmax(self.class_weight - current_weight) + 1
                # class_id = np.random.choice(list(range(1, self.num_class)))
                # class_id = self.calc_sample_class_id()

                if i * num_patches_volume + j >= 0.5 * self.batch_size:
                    force_fg = True
                else:
                    force_fg = False
                data_patch, seg_patch = self.crop_patch(image, segmentation, force_fg)

                data_patches.append(data_patch[None])
                seg_patches.append(seg_patch[None])
                image_names.append(selected_samples[0].split("/")[-1])
                # batch_class_ids += contained_ids
                # for id in contained_ids:
                #     MyDataloader.current_counts[id] += 1

        data_patches = np.array(data_patches).astype(np.float32)
        seg_patches = np.array(seg_patches).astype(np.float32)
        return {'data': data_patches, 'seg': seg_patches, 'image_names': image_names}

    def crop_patch(self, image, label, force_fg):
        shape = image.shape
        crop_size = self.generator_patch_size
        # select the centre of patch
        if not force_fg:
            patch_centre_z = np.random.randint(patch_size[0]//2, shape[0] - patch_size[0]//2)
            patch_centre_x = np.random.randint(patch_size[1]//2, shape[1] - patch_size[1]//2)
            patch_centre_y = np.random.randint(patch_size[2]//2, shape[2] - patch_size[2]//2)
        else:
            # select one class contained in current volume randomly and
            # then pick one voxel belongs to the selected class randomly
            contained_class = np.unique(label[label != 0])
            if self.class_weight is None:
                selected_class = np.random.choice(contained_class)
            else:
                # select one of the top K classes randomly
                wgt_thresh = np.sort(self.class_weight)[int(0.5 * len(self.class_weight))]
                k_high_class = np.argwhere(self.class_weight >= wgt_thresh).squeeze()
                selected_class = np.random.choice(np.intersect1d(contained_class, k_high_class))

            voxels_of_class = np.argwhere(label == selected_class)
            selected_voxel = voxels_of_class[np.random.choice(len(voxels_of_class))]
            patch_centre_z, patch_centre_x, patch_centre_y = selected_voxel

        crop_radius = [i // 2 for i in crop_size]
        slice_z = slice(max(0, patch_centre_z - crop_radius[0]), min(shape[0], patch_centre_z + crop_radius[0]))
        slice_x = slice(max(0, patch_centre_x - crop_radius[1]), min(shape[1], patch_centre_x + crop_radius[1]))
        slice_y = slice(max(0, patch_centre_y - crop_radius[2]), min(shape[2], patch_centre_y + crop_radius[2]))
        data_patch = image[slice_z, slice_x, slice_y]
        seg_patch = label[slice_z, slice_x, slice_y]

        # pad the patch if need
        data_patch = np.pad(data_patch,
                            ((-min(0, patch_centre_z - crop_radius[0]), max(patch_centre_z + crop_radius[0] - shape[0], 0)),
                             (-min(0, patch_centre_x - crop_radius[1]), max(patch_centre_x + crop_radius[1] - shape[1], 0)),
                             (-min(0, patch_centre_y - crop_radius[2]), max(patch_centre_y + crop_radius[2] - shape[2], 0))),
                            mode=data_pad_mode, constant_values=data_pad_val)
        seg_patch = np.pad(seg_patch,
                            ((-min(0, patch_centre_z - crop_radius[0]), max(patch_centre_z + crop_radius[0] - shape[0], 0)),
                             (-min(0, patch_centre_x - crop_radius[1]), max(patch_centre_x + crop_radius[1] - shape[1], 0)),
                             (-min(0, patch_centre_y - crop_radius[2]), max(patch_centre_y + crop_radius[2] - shape[2], 0))),
                            mode='constant', constant_values=seg_pad_val)

        # contained_class_id = [class_id]
        # num_pixel_thresh = 0.1 * np.product(patch_size)
        # for cid in range(0, self.num_class):
        #     if np.sum(seg_patch == cid) > num_pixel_thresh:
        #         contained_class_id.append(cid)

        return data_patch, seg_patch


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
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, patch_center_dist_from_border=None,
            do_elastic_deform=False, deformation_scale=(0.25, 0.5),
            do_rotation=True,
            angle_x=(-rotation_x, rotation_x),
            angle_y=(-rotation_y, rotation_y),
            angle_z=(-rotation_z, rotation_z),
            do_scale=True, scale=range_scale,
            border_mode_data=data_pad_mode, border_cval_data=data_pad_val,
            border_mode_seg='constant', border_cval_seg=seg_pad_val,
            order_seg=1, order_data=3,
            random_crop=False,
            p_el_per_sample=0, p_rot_per_sample=0.2, p_scale_per_sample=0.2
        )
    )

    # now we mirror along all axes
    # tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    
    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, p_per_sample=0.2))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=True, per_channel=True, p_per_sample=0.2))
    
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


def get_data_loader(augmenter_processes=augmenter_processes, dataloader_threads=dataloader_threads, class_weight=None):
    dl = MyDataloader(dataloader_threads, class_weight)
    trans = get_train_transform()

    return MultiThreadedAugmenter(dl, trans, augmenter_processes, num_cached_per_queue=1)


if __name__ == "__main__":
    ml = get_data_loader()

    batches = []
    dataset_info = json.load(open(dataset_info_file, 'r'))
    save_id = 0
    median_spacing = dataset_info['median_spacing']
    for i, batch in enumerate(ml):
        print(i, batch['data'].shape, batch['seg'].shape, batch['image_names'])
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


