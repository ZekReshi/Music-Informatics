import numpy as np

from PM2S.configs import *
from PM2S.data.dataset_base import BaseDataset
from PM2S.constants import *


class TempoDataset(BaseDataset):

    def __init__(self, workspace, split):
        super().__init__(workspace, split)

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        tempo = annotations['tempo']
        tempi = np.zeros(len(note_sequence)).astype(float)

        for i in range(len(note_sequence)):
            onset = note_sequence[i, 1]
            for t in tempo:
                if t[0] > onset + tolerance:
                    break
                tempi[i] = t[1]

        # padding
        length = len(note_sequence)
        if length < max_length:
            note_sequence = np.concatenate([note_sequence, np.zeros((max_length - length, 4))])
            tempi = np.concatenate([tempi, np.zeros((max_length - length))])

        return note_sequence, tempi, length
