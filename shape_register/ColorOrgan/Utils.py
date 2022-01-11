import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.ndimage.filters import gaussian_filter
from scipy import stats, io
import numpy as np
import os
import SimpleITK as sitk


def computeScalarsWithSegError(gt, seg):
    """
    根据分割的误差计算各体素点的标量值
    :param gt: (w, h, z)
    :param seg: 模型预测的softmax置信度volume (c, w, h, z)
    :return:
    """
    label_num = seg.shape[0]
    # convert gt to one-hot
    gt_onehot = (np.arange(label_num) == gt.ravel()[:, None]).astype(np.uint8).reshape(*gt.shape, label_num).transpose(
        (3, 0, 1, 2))
    scalar_volume = np.sum(gt_onehot * seg, axis=0)
    # scalar_volume[gt == 0] = 0
    scalar_volume[(scalar_volume > 0.8) & (gt == 6) ] = 0.8
    return scalar_volume


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


def computeScalarsWithRandomSample(patch_size, gt):
    """
    根据被采样概率计算各体素点的标量值
    :param gt:
    :return:
    """
    sample_num = 500
    shape = gt.shape
    sample_counts = np.zeros((shape[0] + patch_size[0], shape[1] + patch_size[1], shape[1] + patch_size[1]))
    patch_matrix = calc_weight_matrix(patch_size)

    for _ in range(sample_num):
        z = np.random.choice(range(shape[0]))
        x = np.random.choice(range(shape[1]))
        y = np.random.choice(range(shape[2]))

        sample_counts[z:z + patch_size[0], x:x + patch_size[1], y:y + patch_size[2]] += 1

    prob_volume = sample_counts[
                  patch_size[0] // 2:patch_size[0] // 2 + shape[0],
                  patch_size[1] // 2:patch_size[1] // 2 + shape[1],
                  patch_size[2] // 2:patch_size[2] // 2 + shape[2]]
    # prob_volume[gt == 0] = 0
    prob_volume /= prob_volume.max()

    return prob_volume


def computeScalarsWithWeightSample(patch_size, gt, weights):
    """
    根据被采样概率计算各体素点的标量值
    :param gt:
    :return:
    """
    sample_num = 6000
    shape = gt.shape
    sample_counts = np.zeros((shape[0] + patch_size[0], shape[1] + patch_size[1], shape[1] + patch_size[1]))
    patch_matrix = calc_weight_matrix(patch_size)

    probs = np.array(weights) / np.sum(weights)
    classes = 1 + np.arange(len(weights))
    sample_cls_list = np.random.choice(classes, size=sample_num, p=probs)
    cls_sample_sum = [np.sum(sample_cls_list == i) for i in classes]

    for i in range(len(cls_sample_sum)):
        cls_points = np.argwhere(gt == (i + 1))
        cls_points_length = len(cls_points)
        c = 0
        while c < cls_sample_sum[i]:
            z, x, y = cls_points[np.random.choice(np.arange(cls_points_length))]
            sample_counts[z:z + patch_size[0], x:x + patch_size[1], y:y + patch_size[2]] += patch_matrix
            c += 1

    prob_volume = sample_counts[
                  patch_size[0] // 2:patch_size[0] // 2 + shape[0],
                  patch_size[1] // 2:patch_size[1] // 2 + shape[1],
                  patch_size[2] // 2:patch_size[2] // 2 + shape[2]]
    prob_volume[gt == 0] = 0
    prob_volume /= prob_volume.max()

    return prob_volume


def computeScalarsWithErrorSample(patch_size, gt, error):
    sample_num = 6000
    c = 0
    shape = gt.shape
    sample_counts = np.zeros((shape[0] + patch_size[0], shape[1] + patch_size[1], shape[1] + patch_size[1]))
    patch_matrix = calc_weight_matrix(patch_size)

    while c < sample_num:
        z = np.random.choice(range(shape[0]))
        x = np.random.choice(range(shape[1]))
        y = np.random.choice(range(shape[2]))

        if np.random.rand() < error[z, x, y]:
            sample_counts[z:z + patch_size[0], x:x + patch_size[1], y:y + patch_size[2]] += patch_matrix
            c += 1
            # print(c)

    prob_volume = sample_counts[
                  patch_size[0] // 2:patch_size[0] // 2 + shape[0],
                  patch_size[1] // 2:patch_size[1] // 2 + shape[1],
                  patch_size[2] // 2:patch_size[2] // 2 + shape[2]]
    # prob_volume[gt == 0] = 0
    prob_volume = (prob_volume - prob_volume.min()) / (prob_volume.max() - prob_volume.min())
    prob_volume_temp = np.copy(prob_volume)
    # prob_volume += 0.3
    prob_volume[prob_volume_temp < 0.3] += 0.2
    prob_volume[prob_volume_temp >= 0.3] += 0.5

    p1 = np.argwhere(gt == 11)
    p3 = np.argwhere(gt == 12)
    # p3 = np.argwhere(gt == 13)

    pid1 = np.random.choice(np.arange(len(p1)), size=150, replace=False)
    pid3 = np.random.choice(np.arange(len(p3)), size=40, replace=False)
    addition_point = np.concatenate((p1[pid1], p3[pid3]))
    # prob_volume = np.zeros(gt.shape)
    for p in addition_point:
        z, x, y = p
        prob_volume[z - patch_size[0]//12:z + patch_size[0]//12,
                    x - patch_size[1]//12:x + patch_size[1]//12,
                    y - patch_size[2]//12:y + patch_size[2]//12] += 1

    return prob_volume


def getNeighborNonZero(array, z, x, y, r=1):
    elements = array[z - r:z + r + 1, x - r:x + r + 1, y - r:y + r + 1].flatten()
    non_zero_elements = elements[elements != 0]
    while not len(non_zero_elements):
        r += 1
        elements = array[z - r:z + r + 1, x - r:z + r + 1, y - r:y + r + 1].flatten()
        non_zero_elements = elements[elements != 0]

    return stats.mode(non_zero_elements)[0][0]


def getNeighborMeanValue(array, z, x, y, r=1, delete_zero=False):
    elements = array[z - r:z + r + 1, x - r:x + r + 1, y - r:y + r + 1].flatten()
    if not delete_zero:
        return np.mean(elements)
    else:
        non_zero_elements = elements[elements != 0]
        return np.mean(non_zero_elements)


def niiToArray(nii_path):
    # reader = vtk.vtkNIFTIImageReader()
    # reader.SetFileName(nii_path)
    # reader.Update()
    vol = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(vol)
    spacing = vol.GetSpacing()
    return array, spacing


def arrayToPoly(gt_array, scalar_array, spacing, lbl_idx, topology_off=True, filename=None, delete_zero=False):
    array = gt_array.copy()
    array[array != lbl_idx] = 0
    # array to vtkImageData
    flatten_array = array.ravel()
    shape = np.array(array.shape)
    vtk_data_array = numpy_to_vtk(
        num_array=flatten_array,  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_FLOAT)

    # Convert the VTK array to vtkImageData
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(shape[::-1])
    img_vtk.SetSpacing(spacing)
    img_vtk.GetPointData().SetScalars(vtk_data_array)

    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputData(img_vtk)
    surf.SetValue(0, lbl_idx)  # use surf.GenerateValues function if more than one contour is available in the file
    # surf.GenerateValues(13, 0, 1)  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(50)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()

    # decimation
    if topology_off is not None:
        numPts = smoother.GetOutput().GetNumberOfPoints()
        reduction = 0
        if numPts > 1000:
            reduction = (numPts - 1000) / numPts

        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(smoother.GetOutput())
        decimate.SetTargetReduction(reduction)
        if topology_off:
            decimate.PreserveTopologyOff()
        else:
            decimate.PreserveTopologyOn()
        decimate.Update()

        poly = decimate.GetOutput()  # polydata
        numPts = poly.GetNumberOfPoints()
        print("reduction is : ", reduction, " %d points in ploy for label %d" % (numPts, lbl_idx))
    else:
        poly = smoother.GetOutput()
        numPts = poly.GetNumberOfPoints()
        print(" %d points in ploy for label %d" % (numPts, lbl_idx))

    # set scalar value for poly
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfValues(numPts)

    coords = []
    values = set()
    for i in range(numPts):
        poly_coord = [c for c in [poly.GetPoint(i)[0], poly.GetPoint(i)[1], poly.GetPoint(i)[2]]]
        coord = [round(c / s) for c, s in zip(poly_coord, spacing)]
        coords.append([coord[2], coord[1], coord[0]])  # z, x, y
        # s = getNeighborNonZero(scalar_array, coord[2], coord[1], coord[0])
        s = getNeighborMeanValue(scalar_array, coord[2], coord[1], coord[0], r=1, delete_zero=delete_zero)
        values.add(s)
        scalars.SetValue(i, float(s))

    # save normalized coordinate of points in polydata
    if filename is not None:
        coords = np.array(coords)
        coords = coords / shape
        io.savemat(os.path.join(landmarks_path, "%s_%s.mat" % (filename.split(".")[0], lbl_idx)), {"x": coords})

    poly.GetPointData().SetScalars(scalars)
    return poly


def makeGlyphActor(src):
    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(16)
    arrow.SetTipLength(0.3)
    arrow.SetTipRadius(0.1)

    # Update normals on newly smoothed polydata
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(src)
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ComputeCellNormalsOn()
    normalGenerator.AutoOrientNormalsOn()
    normalGenerator.Update()

    centers = vtk.vtkCellCenters()
    centers.SetInputConnection(normalGenerator.GetOutputPort())
    centers.Update()

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputData(centers.GetOutput())
    glyph.SetVectorModeToUseNormal()
    glyph.SetScaleFactor(6)
    glyph.SetColorModeToColorByVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.OrientOn()
    glyph.Update()

    glyphMapper = vtk.vtkPolyDataMapper()
    glyphMapper.SetInputConnection(glyph.GetOutputPort())
    # glyphMapper.SetScalarModeToUsePointFieldData()
    # glyphMapper.SetColorModeToMapScalars()
    # glyphMapper.ScalarVisibilityOn()
    # glyphMapper.SelectColorArray('Elevation')
    # Colour by scalars.
    # glyphMapper.SetScalarRange(0, 1)

    glyphActor = vtk.vtkActor()
    glyphActor.SetMapper(glyphMapper)

    return glyphActor


def showPolyDatas(poly_list):
    # Create a lookup table to share between the mapper and the scalarbar
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(0, 1)
    lut.SetHueRange(0.7, 0.0)
    lut.SetSaturationRange(1, 1)
    lut.SetValueRange(1, 1)
    lut.Build()

    ren = vtk.vtkRenderer()
    for poly in poly_list:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        glyph_actor = makeGlyphActor(poly)

        ren.AddActor(actor)
        # ren.AddViewProp(glyph_actor)

    scalarBarActor = vtk.vtkScalarBarActor()
    scalarBarActor.SetTitle("Color")
    scalarBarActor.SetNumberOfLabels(4)
    scalarBarActor.SetLookupTable(lut)

    ren.ResetCamera()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.AddActor2D(scalarBarActor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(900, 900)
    renWin.Render()

    camera = ren.GetActiveCamera()
    camera.Elevation(-100)
    ren.SetActiveCamera(camera)
    renWin.Render()

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


if __name__ == '__main__':
    labels_path = "../Data/labels/"
    landmarks_path = "../Data/landmarks/"
    patch_size = [24, 48, 48]

    # for label in os.listdir(labels_path):
    #     gt_volume, spacing = niiToArray(os.path.join(labels_path, label))
    #     print(label, gt_volume.shape)

    # 提取关键点
    # for label in os.listdir(labels_path):
    #     gt_volume, spacing = niiToArray(os.path.join(labels_path, label))
    #     scalars_volume = gt_volume / gt_volume.max()
    #     # scalars_volume = computeScalarsWithRandomSample(patch_size, gt_volume)
    #     # scalars_volume = computeScalarsWithWeightSample(patch_size, gt_volume, weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    #
    #     print(label, scalars_volume.shape, scalars_volume.max(), scalars_volume.min())
    #
    #     poly_list = []
    #     for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    #         poly = arrayToPoly(gt_volume, scalars_volume, spacing, i, topology_off=True, delete_zero=True)
    #         poly_list.append(poly)
    #     showPolyDatas(poly_list)

    # 根据predcitino显示误差图
    # gt_volume, spacing = niiToArray(os.path.join(labels_path, "img0008.nii.gz"))
    # gt_volume = gt_volume[1:, :, :]
    # # seg_volume = np.load("../Data/img0001-fullres.npz")["softmax"]
    # # scalars_volume = computeScalarsWithSegError(gt_volume, seg_volume)
    # scalars_volume = computeScalarsWithWeightSample(patch_size, gt_volume,
    #                                                 weights=[20, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # print(scalars_volume.shape, scalars_volume.max(), scalars_volume.min())
    #
    # poly_list = []
    # # for i in [4, 5, 7, 9, 10, 11, 12, 13]:
    # for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    #     poly = arrayToPoly(gt_volume, scalars_volume, spacing, i, topology_off=False, filename=None, delete_zero=True)
    #     poly_list.append(poly)
    # showPolyDatas(poly_list)

    # 0008, 0010: (148,512,512)  0001: (147,512,512)  0009: 149
    gt_volume, spacing = niiToArray(os.path.join(labels_path, "img0001.nii.gz"))
    seg_volume = np.load("../Data/img0001-fullres.npz")["softmax"]
    error = 1 - computeScalarsWithSegError(gt_volume, seg_volume)
    scalars_volume = computeScalarsWithErrorSample(patch_size, gt_volume, error)
    print(scalars_volume.shape, scalars_volume.max(), scalars_volume.min())

    poly_list = []
    # for i in [4, 5, 7, 9, 10, 11, 12, 13]:
    for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        poly = arrayToPoly(gt_volume, scalars_volume, spacing, i, topology_off=False, filename=None, delete_zero=True)
        poly_list.append(poly)
    showPolyDatas(poly_list)
