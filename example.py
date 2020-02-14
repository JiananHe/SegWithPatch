import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
from skimage import measure
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#
# def elastic_transform(image, alpha, sigma):
#     """
#     Elastic deformation of images as described in [1].
#
#     [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
#         Neural Networks applied to Visual Document Analysis", in Proc. of the
#         International Conference on Document Analysis and Recognition, 2003.
#
#     Based on gist https://gist.github.com/erniejunior/601cdf56d2b424757de5
#
#     Args:
#         image (np.ndarray): image to be deformed
#         alpha (list): scale of transformation for each dimension, where larger
#             values have more deformation
#         sigma (list): Gaussian window of deformation for each dimension, where
#             smaller values have more localised deformation
#
#     Returns:
#         np.ndarray: deformed image
#     """
#
#     assert len(alpha) == len(sigma), \
#         "Dimensions of alpha and sigma are different"
#
#     channelbool = image.ndim - len(alpha)
#     out = np.zeros((len(alpha) + channelbool, ) + image.shape)
#
#     # Generate a Gaussian filter, leaving channel dimensions zeroes
#     for jj in range(len(alpha)):
#         array = (np.random.rand(*image.shape) * 2 - 1)
#         out[jj] = gaussian_filter(array, sigma[jj],
#                                   mode="constant", cval=0) * alpha[jj]
#
#     # Map mask to indices
#     shapes = list(map(lambda x: slice(0, x, None), image.shape))
#     grid = np.broadcast_arrays(*np.ogrid[shapes])
#     indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))
#
#     # Transform image based on masked indices
#     transformed_image = map_coordinates(image, indices, order=0,
#                                         mode='reflect').reshape(image.shape)
#
#     return transformed_image
#
#
# vol = sitk.ReadImage(r'D:\Projects\OrgansSegment\SegWithPatch\samples\Training\img\img0002.nii.gz')
# arr = sitk.GetArrayFromImage(vol)
# trans_arr = elastic_transform(arr, [100]*3, [8]*3)
# trans_vol = sitk.GetImageFromArray(trans_arr)
# trans_vol.SetSpacing(vol.GetSpacing())
# trans_vol.SetOrigin(vol.GetOrigin())
# trans_vol.SetDirection(vol.GetDirection())
# sitk.WriteImage(trans_vol, r'D:\test.nii.gz')


# 最大连通区域
ct_upper = 275.0
ct_lower = -125.0
new_spacing = [0.758, 0.758, 3.0]
seg_raw_vol = sitk.ReadImage(r"D:\Projects\OrgansSegment\BTCV\RawData\Training\img\img0002.nii.gz")
img_array = sitk.GetArrayFromImage(seg_raw_vol)
raw_shape = img_array.shape

# 阈值截取
img_array = np.clip(img_array, ct_lower, ct_upper).astype(np.float32)

region_array = np.zeros(img_array.shape)
region_array[img_array>ct_lower] = 1

labels, num = measure.label(region_array, return_num=True)
regions = measure.regionprops(labels)
regions_area = [regions[i].area for i in range(num)]
region_num = regions_area.index(max(regions_area))
bbox = regions[region_num].bbox
img_array = img_array[bbox[0]:bbox[3]+1, bbox[1]:bbox[4]+1, bbox[2]:bbox[5]+1]

# 重采样, lbl应使用最近邻插值
img_spacing = seg_raw_vol.GetSpacing()
img_array = ndimage.zoom(img_array, (img_spacing[2] / new_spacing[2],
                                     img_spacing[0] / new_spacing[0],
                                     img_spacing[1] / new_spacing[1]), order=3)

#
# bbox_array = np.zeros(region_array.shape)
# bbox_array[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1] = 1
# cv2.imshow("bbox", bbox_array)
# cv2.waitKey(0)

new_img_vol = sitk.GetImageFromArray(img_array)
new_img_vol.SetDirection(seg_raw_vol.GetDirection())
new_img_vol.SetOrigin(seg_raw_vol.GetOrigin())
new_img_vol.SetSpacing(seg_raw_vol.GetSpacing())

sitk.WriteImage(new_img_vol, r"D:\\test.nii.gz")