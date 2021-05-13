import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    ToTensor)

from ..config import get_output_dir
from .mot_sequence import MOTSequence
import random


class MOTreID(MOTSequence):
    """Multiple Object Tracking Dataset.

    This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

    Values for P are normally 18 and K 4
    """

    def __init__(self, seq_name, mot_dir, vis_threshold, P, K, max_per_person, crop_H, crop_W,
                transform, normalize_mean=None, normalize_std=None, logger=print):
        super().__init__(seq_name, mot_dir, vis_threshold=vis_threshold)

        self.P = P
        self.K = K
        self.max_per_person = max_per_person
        self.crop_H = crop_H
        self.crop_W = crop_W
        self.logger = logger
        # self.normalize_mean, self.normalize_std = normalize_mean, normalize_std

        self.neg_crop_transform = Compose([
            RandomCrop((288, 144)),
            RandomHorizontalFlip()])

        if transform == "random":
            self.transform = Compose([
                RandomCrop((crop_H, crop_W)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(normalize_mean, normalize_std)])
        elif transform == "center":
            self.transform = Compose([
                CenterCrop((crop_H, crop_W)),
                ToTensor(),
                Normalize(normalize_mean, normalize_std)])
        else:
            raise NotImplementedError("Tranformation not understood: {}".format(transform))

        # self.data = self.build_samples()
        self.data, self.neg_samples = self.build_samples()

    def __getitem__(self, idx):
        """Return the ith triplet"""

        res = []
        # idx belongs to the positive sampled person
        pos = self.data[idx]
        res.append(pos[np.random.choice(pos.shape[0], self.K, replace=False)])

        # exclude idx here
        random_sampled = False
        try:
            neg_indices = np.random.choice([
                i for i in range(len(self.data))
                if i != idx], self.P-1, replace=False)
            for i in neg_indices:
                neg = self.data[i]
                res.append(neg[np.random.choice(neg.shape[0], self.K, replace=False)])
        except (IndexError, ValueError): # <=1 sample -> no neg sample -> generate random non overlapping sample as negative
            # neg_indices = np.random.choice([
            #     i for i in range(len(self.neg_samples))], self.P-1, replace=False)
            # for i in neg_indices:
            #     neg = self.neg_samples[i]
            #     res.append(neg[np.random.choice(neg.shape[0], self.K, replace=False)])
            idx_to_sample = [i for i in range(len(self.neg_samples))]
            num_to_sample = res[0].shape[0]
            neg_indices = random.sample(idx_to_sample, num_to_sample)
            res.append(np.array([self.neg_samples[i] for i in neg_indices])) # use as many negative samples as we have positive samples
            random_sampled = True
        # concatenate the results
        r = []
        for pers in res:
            for im in pers:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im)
                r.append(self.transform(im))
        images = torch.stack(r, 0)

        # construct the labels
        labels = [idx] * self.K

        if not random_sampled: # normaly way
            for l in neg_indices:
                labels += [l] * self.K
        else:
            labels += [0] * self.K  # TODO does 0 == None class?

        labels = np.array(labels)

        batch = [images, labels]

        return batch

    def build_samples(self):
        """Builds the samples out of the sequence."""

        tracks = {}

        for sample in self.data:
            im_path = sample['im_path']
            gt = sample['gt']

            for k,v in tracks.items():
                if k in gt.keys():
                    v.append({'id':k, 'im_path':im_path, 'gt':gt[k]})
                    del gt[k]

            # For all remaining BB in gt new tracks are created
            for k,v in gt.items():
                tracks[k] = [{'id':k, 'im_path':im_path, 'gt':v}]

        # sample max_per_person images and filter out tracks smaller than 4 samples
        #outdir = get_output_dir("siamese_test")
        res = []
        neg_samples = []
        for k,v in tracks.items():
            l = len(v)
            # raise AttributeError(k,v, self.K, len(v))
            if l >= self.K:
                pers = []
                if l > self.max_per_person:
                    for i in np.random.choice(l, self.max_per_person, replace=False):
                        try:
                            pers.append(self.build_crop(v[i]['im_path'], v[i]['gt']))
                            if len(neg_samples) < self.P:
                                neg_samples.append(np.array(self.neg_crop_transform(Image.fromarray(cv2.imread(v[i]['im_path'])))))
                        except cv2.error:
                            print(v[i]['im_path'])
                            continue
                else:
                    for i in range(l):
                        pers.append(self.build_crop(v[i]['im_path'], v[i]['gt']))
                        if len(neg_samples) < self.P:
                            neg_samples.append(np.array(self.neg_crop_transform(Image.fromarray(cv2.imread(v[i]['im_path'])))))

                #for i,v in enumerate(pers):
                #	cv2.imwrite(osp.join(outdir, str(k)+'_'+str(i)+'.png'),v)
                res.append(np.array(pers))


        if self._seq_name:
            self.logger(f"[*] Loaded {len(res)} persons from sequence {self._seq_name}.")

        # return res
        return res, neg_samples

    def build_crop(self, im_path, gt):
        im = cv2.imread(im_path)
        height, width, _ = im.shape
        #blobs, im_scales = _get_blobs(im)
        #im = blobs['data'][0]
        #gt = gt * im_scales[0]
        # clip to image boundary
        w = gt[2] - gt[0]
        h = gt[3] - gt[1]
        context = 0
        gt[0] = np.clip(gt[0]-context*w, 0, width-1)
        gt[1] = np.clip(gt[1]-context*h, 0, height-1)
        gt[2] = np.clip(gt[2]+context*w, 0, width-1)
        gt[3] = np.clip(gt[3]+context*h, 0, height-1)

        # assume doesn't hit edge of image
        if w == 0:
            gt[2] += 2
        if h == 0:
            gt[3] += 2
        im = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

        try:
            im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            print(im, im_path)
            raise cv2.error()

        return im
