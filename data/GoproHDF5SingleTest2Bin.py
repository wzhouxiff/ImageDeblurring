## simulated dataset containing ground-truth deblurred.
# Event3.mat
import os
from data.dataset import DatasetBase
from utils import cv_utils
import numpy as np
import zlib
import scipy.io as scio
import lmdb
import csv
import cv2
import random
import h5py

class GoProHDF5(DatasetBase):
    def __init__(self, opt, is_for_train=False):
        super(GoProHDF5, self).__init__(opt, is_for_train)
        self._name = 'GoProHDF5'
        print('Loading GoProHDF5 dataset...')
        self._opt = opt
        self.inter_num = self._opt.inter_num
        self.H = 720
        self.W = 1280
        self.is_for_train = is_for_train
        self._read_dataset_paths()

    def _event_cumulate(self, events):
        c, h, w = events.shape
        image_pos = np.zeros((h,w), dtype=np.uint8)
        image_neg = np.zeros((h,w), dtype=np.uint8)

        for i in range(c):
            pos = np.where(events[i]>0)
            image_pos[pos] += events[i][pos].astype( np.uint8)

            neg = np.where(events[i]<0)
            image_neg[neg] += (-events[i][neg]).astype(np.uint8)

        # np.zeros((h,w), dtype=np.uint8)
        image_rgb = np.stack([image_pos, image_neg], axis=0).astype(np.float32) * 50. / 255.

        return image_rgb

    def _read_dataset_paths(self):
        self.root = os.path.expanduser(self._opt.data_dir)
        if self.root[-1] == '/':
            self.root = self.root[:-1]

        self.root = os.path.join(self._opt.data_dir, 'test')

        with open(os.path.join(self._opt.data_dir, 'test.txt')) as f:
            self.name_list = [name.strip() for name in f.readlines()]
        self.name_list = [self.name_list[i][:-1] for i in range(0, len(self.name_list), 4)]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        
        sample = {}

        blurs = []
        for i in range(1,5):
            with h5py.File(os.path.join(self.root, self.name_list[index]+str(i)+'_blur.h5'), 'r') as f:
                blur = np.array(list(f['data']))
            blur = blur.astype(np.float32) / 255.
            blurs.append(blur)
        sample['blurred'] = np.concatenate([np.concatenate([blurs[0], blurs[1]], axis=-1),
                                                np.concatenate([blurs[2], blurs[3]], axis=-1)], axis=-2)

        gts = []
        for i in range(1,5):
            with h5py.File(os.path.join(self.root, self.name_list[index]+str(i)+'_gt.h5'), 'r') as f:
                gt = np.array(list(f['data']))
            gt = gt.astype(np.float32) / 255.
            gts.append(gt)
        sample['gts'] = np.concatenate([np.concatenate([gts[0], gts[1]], axis=-1),
                                            np.concatenate([gts[2], gts[3]], axis=-1)], axis=-2)

        big_events = []
        for i in range(1,5):
            with h5py.File(os.path.join(self.root, self.name_list[index]+str(i)+'_events.h5'), 'r') as f:
                events = np.array(list(f['data']))
            events = events.astype(np.float32)
            # events = (events - 127) / 20.
            events = events * 50./ 255.
            big_events.append(events)
        events = np.concatenate([np.concatenate([big_events[0], big_events[1]], axis=-1),
                                            np.concatenate([big_events[2], big_events[3]], axis=-1)], axis=-2)

        sample['events'] = events

        sample['dataname'] = self.name_list[index]

        return sample
