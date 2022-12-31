import os, glob
import torch
import numpy as np
import scipy.io
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
import cv2
import h5py
from create_associations import associate


def convert_label(label):

    onehot = np.zeros((1, 50, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


class BSDS:
    def __init__(self, root, features, split="train", color_transforms=None, geo_transforms=None):
        self.gt_dir = os.path.join(root, "BSDS500/data/groundTruth", split)
        self.img_dir = os.path.join(root, "BSDS500/data/images", split)

        self.index = os.listdir(self.gt_dir)

        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms
        self._load_features(features)


    def __getitem__(self, idx):
        associations = associate('/home/asl-student/Desktop/Smazzucco/ssn-pytorch/BSR/BSDS500/data/images/train')
        idx = self.index[idx][:-4]
        gt = scipy.io.loadmat(os.path.join(self.gt_dir, idx+".mat"))
        t = np.random.randint(0, len(gt['groundTruth'][0]))
        gt = gt['groundTruth'][0][t][0][0][0]

        if gt.shape[0] > gt.shape[1]:
            gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img = rgb2lab(plt.imread(os.path.join(self.img_dir, idx+".jpg")))

        gt = gt.astype(np.int64)
        img = img.astype(np.float32)

        if self.color_transforms is not None:
            img = self.color_transforms(img)

        if self.geo_transforms is not None:
            img, gt = self.geo_transforms([img, gt])

        gt = convert_label(gt)
        gt = torch.from_numpy(gt)

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        img = self.ssn_features[associations[idx]]
        return img, gt.reshape(50, -1).float()

    def _load_features(self, features):
        with h5py.File(os.path.join('/home/asl-student/Desktop/Smazzucco/ssn-pytorch/BSR/BSDS500/data/images/', 'features.hdf'),
                       'r') as hdf:
            features = hdf[f'features/{features}'][:]
            N, H, W, C = features.shape
            self.ssn_features = features.reshape(N, C, H, W)
            self.ssn_features = torch.nn.Upsample(
                (200, 200), mode='nearest')(torch.Tensor(self.ssn_features))

    def __len__(self):
        return len(self.index)
