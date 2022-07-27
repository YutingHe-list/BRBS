from os.path import join
from os import listdir
from scipy.io import loadmat
import SimpleITK as sitk
import pandas as pd
from torch.utils import data
import numpy as np

# from utils.augmentation_cpu import MirrorTransform, SpatialTransform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir, num_classes, shot=5):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.labeled_file_dir = labeled_file_dir
        self.num_classes = num_classes
        self.labeled_filenames = self.labeled_filenames[shot:]
        print(self.labeled_filenames)

    def __getitem__(self, index):
        labed_img = sitk.ReadImage(join(self.labeled_file_dir, 'image', self.labeled_filenames[index]))
        labed_img = sitk.GetArrayFromImage(labed_img)
        labed_img = np.where(labed_img < 0., 0., labed_img)
        labed_img = np.where(labed_img > 2048., 2048., labed_img)
        labed_img = labed_img / 2048.
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :, :]
        labed_lab = sitk.ReadImage(join(self.labeled_file_dir, 'label', self.labeled_filenames[index]))
        labed_lab = sitk.GetArrayFromImage(labed_lab)
        labed_lab = np.where(labed_lab == 205, 1, labed_lab)
        labed_lab = np.where(labed_lab == 420, 2, labed_lab)
        labed_lab = np.where(labed_lab == 500, 3, labed_lab)
        labed_lab = np.where(labed_lab == 550, 4, labed_lab)
        labed_lab = np.where(labed_lab == 600, 5, labed_lab)
        labed_lab = np.where(labed_lab == 820, 6, labed_lab)
        labed_lab = np.where(labed_lab == 850, 7, labed_lab)
        labed_lab = self.to_categorical(labed_lab, self.num_classes)
        labed_lab = labed_lab.astype(np.float32)

        return labed_img, labed_lab, self.labeled_filenames[index]

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.labeled_filenames)

