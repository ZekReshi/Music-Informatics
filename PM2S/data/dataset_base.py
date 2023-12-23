import os, sys

from PM2S.constants import keyName2Number

sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch
import pandas as pd
from collections import defaultdict
import random
from pathlib import Path
import partitura as pt
import numpy as np

from PM2S.configs import *
from PM2S.data.data_augmentation import DataAugmentation


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, workspace, split):

        # parameters
        self.workspace = workspace
        self.feature_folder = os.path.join(workspace, 'Dataset', 'train')
        self.split = split

        # Get metadata by split
        self.metadata = pd.read_csv(os.path.join(self.feature_folder, 'key-meter_train_gt.csv'), delimiter=',')
        self.metadata.reset_index(inplace=True)

        # Get distinct pieces
        self.piece2row = defaultdict(list)
        for i, row in self.metadata.iterrows():
            self.piece2row[row['filename']].append(i)
        self.pieces = list(self.piece2row.keys())

        # Initialise data augmentation
        self.dataaug = DataAugmentation(extra_note_prob=0, missing_note_prob=0)

    def __len__(self):
        if self.split == 'train' or self.split == 'all':
            # constantly update 200 steps per epoch, not related to training dataset size
            return batch_size * 200

        elif self.split == 'valid':
            # by istinct pieces in validation set
            return batch_size * len(self.piece2row) // 10  # valid dataset size

        elif self.split == 'test':
            return len(self.metadata)

    def _sample_row(self, idx):
        # Sample one row from the metadata
        if self.split == 'train' or self.split == 'all':
            piece_id = random.choice(list(self.piece2row.keys()))   # random sampling by piece
            row_id = random.choice(self.piece2row[piece_id])
        elif self.split == 'valid':
            piece_id = self.pieces[idx // batch_size]    # by istinct pieces in validation set
            row_id = self.piece2row[piece_id][idx % batch_size % len(self.piece2row[piece_id])]
        elif self.split == 'test':
            row_id = idx
        row = self.metadata.iloc[row_id]

        return row

    def _load_data(self, row):
        # Get feature
        p = pt.load_performance_midi(str(Path(self.feature_folder, row['filename'])))
        note_array = p.note_array()
        note_sequence = np.array(list(zip(note_array['pitch'], note_array['onset_sec'], note_array['duration_sec'], note_array['velocity'])))
        annotations = {
            'time_signatures': np.array([(0., row['ts_num'])]),
            'key_signatures': np.array([(0., keyName2Number[row['key']])])
        }

        # Data augmentation
        if self.split == 'train' or self.split == 'all':
            note_sequence, annotations = self.dataaug(note_sequence, annotations)

        # Randomly sample a segment that is at most max_length long
        if self.split == 'train' or self.split == 'all':
            start_idx = random.randint(0, len(note_sequence)-1)
            end_idx = start_idx + max_length
        elif self.split == 'valid':
            start_idx, end_idx = 0, max_length  # validate on the segment starting with the first note
        elif self.split == 'test':
            start_idx, end_idx = 0, len(note_sequence)  # test on the whole note sequence

        if end_idx > len(note_sequence):
            end_idx = len(note_sequence)

        note_sequence = note_sequence[start_idx:end_idx]
        #for key in annotations.keys():
        #    if key in ['onsets_musical', 'note_value', 'hands', 'hands_mask'] and annotations[key] is not None:
        #        annotations[key] = annotations[key][start_idx:end_idx]

        return note_sequence, annotations
