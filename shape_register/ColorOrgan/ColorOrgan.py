import vtk
import numpy as np
import SimpleITK as sitk
from Utils import *




if __name__ == "__main__":
    scalar_volume = computeScalarsWithSampleProb(np.zeros((20, 50, 50)))
    print(scalar_volume.shape)
