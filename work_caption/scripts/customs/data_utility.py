import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchvision

from fairseq.data import FairseqDataset, data_utils
from torch.utils.data.dataloader import default_collate

from PIL import Image

import sentencepiece as sp


def split_file(split):
    return os.path.join('splits', f'karpathy_{split}_images.txt')


def read_split_image_ids_and_paths(split):
    split_df = pd.read_csv(split_file(split), sep=' ', header=None)
    return split_df.iloc[:,1].to_numpy(), split_df.iloc[:,0].to_numpy()


def read_split_image_ids(split):
    return read_split_image_ids_and_paths(split)[0]


def read_image_ids(file, non_redundant=False):
    with open(file, 'r') as f:
        image_ids = [line.strip('\n') for line in f]

    if non_redundant:
        return list(set(image_ids))
    else:
        return image_ids


def read_image_metadata(file):
    df = pd.read_csv(file)
    md = {}

    for img_id, img_h, img_w, num_boxes in zip(df['image_id'], df['image_h'], df['image_w'], df['num_boxes']):
        md[img_id] = {
            'image_h': np.float32(img_h),
            'image_w': np.float32(img_w),
            'num_boxes': num_boxes
        }

    return md


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, num_tokens):
        self.image_paths = image_paths
        self.num_tokens = np.ones(len(self.image_paths), dtype=np.int) * num_tokens

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img=Image.open(image_path)
        img=torchvision.transforms.functional.resize(img, size=[256,256])
        image_tensor=torchvision.transforms.functional.to_tensor(img)
        return torch.FloatTensor(image_tensor)

    @property
    def sizes(self):
        return self.num_tokens

    def collater(self, samples):
        return samples

class CaptionDataset(FairseqDataset):
    def __init__(self, captions_path, sentencepiece_model):
        with open(captions_path) as f:
            lines = f.readlines()
            self.captions = [l.strip('\n') for l in lines]
            self.num_tokens = np.zeros(len(self.captions), dtype=np.int)
            self.sentencepiece_model = sp.SentencePieceProcessor()
            self.sentencepiece_model.Load(sentencepiece_model)

            for idx, c in enumerate(self.captions):
                self.num_tokens[idx] = len(c)

    def __getitem__(self, index):
        return torch.FloatTensor(self.sentencepiece_model.EncodeAsIds(self.captions[index]))

    def __len__(self):
        return len(self.captions)

    @property
    def sizes(self):
        return self.num_tokens

    def collater(self, samples):
        return samples


class ImageCaptionDataset(FairseqDataset):
    def __init__(self, img_ds, cap_ds, cap_dict, shuffle=False):
        self.img_ds = img_ds
        self.cap_ds = cap_ds
        self.cap_dict = cap_dict
        self.shuffle = shuffle

    def __getitem__(self, index):
        source_image = self.img_ds[index]
        target = self.cap_ds[index]

        return {
            'id': index,
            'source_image': source_image,
            'target': target
        }

    def __len__(self):
        return len(self.img_ds)

    def num_tokens(self, index):
        return self.size(index)[1]

    def size(self, index):
        # number of image feature vectors, number of tokens in caption
        return self.img_ds.sizes[index], self.cap_ds.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        # Inspired by LanguagePairDataset.ordered_indices
        indices = indices[np.argsort(self.cap_ds.sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.img_ds.sizes[indices], kind='mergesort')]

    def collater(self, samples):
        indices = []

        source_image_samples = []
        source_lengths = []

        target_samples = []
        target_ntokens = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)

            source_image_samples.append(sample['source_image'])
            source_lengths.append(self.img_ds.sizes[index])

            target_samples.append(sample['target'])
            target_ntokens += self.cap_ds.sizes[index]

        num_sentences = len(samples)

        # FIXME: workaround for edge case in parallel processing
        # (framework passes empty samples list
        # to collater under certain conditions)
        if num_sentences == 0:
            return None

        indices = torch.tensor(indices, dtype=torch.long)

        source_image_batch = self.img_ds.collater(list(source_image_samples))

        # TODO: switch depending on SCST or CE training
        target_batch = data_utils.collate_tokens(target_samples,
                                                     pad_idx=self.cap_dict.pad(),
                                                     eos_idx=self.cap_dict.eos(),
                                                     move_eos_to_beginning=False)
        rotate_batch = data_utils.collate_tokens(target_samples,
                                                     pad_idx=self.cap_dict.pad(),
                                                     eos_idx=self.cap_dict.eos(),
                                                     move_eos_to_beginning=True)

        return {
            'id': indices,
            'net_input': {
                'src_tokens': torch.stack(source_image_batch, dim=0),
                'src_lengths': source_lengths,
                'prev_output_tokens': rotate_batch,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }