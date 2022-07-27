from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir, unlabeled_file_dir, num_classes, shot=5):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.unlabeled_filenames = [x for x in listdir(join(unlabeled_file_dir, 'image')) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir
        self.labeled_file_dir = labeled_file_dir
        self.num_classes = num_classes
        self.labeled_filenames = self.labeled_filenames[:shot]

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.labeled_filenames))
        labed_img = sitk.ReadImage(join(self.labeled_file_dir, 'image', self.labeled_filenames[random_index]))
        labed_img = sitk.GetArrayFromImage(labed_img)
        labed_img = np.where(labed_img < 0., 0., labed_img)
        labed_img = np.where(labed_img > 2048., 2048., labed_img)
        labed_img = labed_img / 2048.
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :, :]
        labed_lab = sitk.ReadImage(join(self.labeled_file_dir, 'label', self.labeled_filenames[random_index]))
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

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img1 = sitk.ReadImage(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]))
        unlabed_img1 = sitk.GetArrayFromImage(unlabed_img1)
        unlabed_img1 = np.where(unlabed_img1 < 0., 0., unlabed_img1)
        unlabed_img1 = np.where(unlabed_img1 > 2048., 2048., unlabed_img1)
        unlabed_img1 = unlabed_img1 / 2048.
        unlabed_img1 = unlabed_img1.astype(np.float32)
        unlabed_img1 = unlabed_img1[np.newaxis, :, :, :]

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img2 = sitk.ReadImage(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]))
        unlabed_img2 = sitk.GetArrayFromImage(unlabed_img2)
        unlabed_img2 = np.where(unlabed_img2 < 0., 0., unlabed_img2)
        unlabed_img2 = np.where(unlabed_img2 > 2048., 2048., unlabed_img2)
        unlabed_img2 = unlabed_img2 / 2048.
        unlabed_img2 = unlabed_img2.astype(np.float32)
        unlabed_img2 = unlabed_img2[np.newaxis, :, :, :]

        return labed_img, labed_lab, unlabed_img1, unlabed_img2

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
        return len(self.unlabeled_filenames)+len(self.labeled_filenames)

