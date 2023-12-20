import numpy as np

from PM2S.configs import *
from PM2S.data.dataset_base import BaseDataset
from PM2S.constants import *


class TimeSignatureDataset(BaseDataset):

    def __init__(self, workspace, split):
        super().__init__(workspace, split)

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        time_signatures = annotations['time_signatures']
        ts_numerators = np.zeros(len(note_sequence)).astype(float)

        for i in range(len(note_sequence)):
            onset = note_sequence[i, 1]
            for ts in time_signatures:
                if ts[0] > onset + tolerance:
                    break
                ts_numerators[i] = tsNume2Index[int(ts[1])] if int(ts[1]) in tsNume2Index.keys() else 0

        # padding
        length = len(note_sequence)
        if length < max_length:
            note_sequence = np.concatenate([note_sequence, np.zeros((max_length - length, 4))])
            ts_numerators = np.concatenate([ts_numerators, np.zeros(max_length - length)])

        return note_sequence, ts_numerators, length

