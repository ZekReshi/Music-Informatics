import argparse

import torch

from PM2S.features._processor import MIDIProcessor
from PM2S.models.key_signature import RNNKeySignatureModel
from PM2S.constants import keyNumber2Name


class RNNKeySignatureProcessor(MIDIProcessor):

    def __init__(self, model_state_dict_path='_model_state_dicts/key_signature/RNNKeySignatureModel.pth', **kwargs):
        super().__init__(model_state_dict_path, **kwargs)

    def load(self, state_dict_path):
        if state_dict_path:
            self._model = RNNKeySignatureModel()
            self._model.load_state_dict(torch.load(state_dict_path))
        else:
            self._model = RNNKeySignatureModel()

    def process(self, note_seq, **kwargs):
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        key_probs = self._model(x)

        # Post-processing
        key_idx = key_probs[0].topk(1, dim=0)[1].squeeze(0).cpu().detach().numpy() # (seq_len,)

        onsets = note_seq[:, 1]
        key_signature_changes = self.pps(key_idx, onsets)

        return key_signature_changes

    def pps(self, key_idx, onsets):
        ks_prev = '0'
        ks_changes = []
        for i in range(len(key_idx)):
            ks_cur = keyNumber2Name[key_idx[i]]
            if i == 0 or ks_cur != ks_prev:
                onset_cur = onsets[i]
                ks_changes.append((onset_cur, ks_cur))
                ks_prev = ks_cur
        return ks_changes
    

if __name__ == '__main__':
    import glob
    import os
    import numpy as np
    import partitura as pt
    from pathlib import Path
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Predict key signatures.')

    parser.add_argument('--datadir', type=str, help='Dataset directory.')
    parser.add_argument('--modeldir', type=str, help='Model directory.')
    parser.add_argument('--outfile', type=str, help='Output file.')

    args = parser.parse_args()

    midi_files = glob.glob(os.path.join(args.datadir, "*.mid"))
    midi_files.sort()

    # Create time and key processors
    processor_key_sig = RNNKeySignatureProcessor(os.path.join(args.modeldir, '_model_state_dicts', 'key_signature', 'RNNKeySignatureModel.pth'))

    results = []
    # Prediction
    for idx, file in enumerate(midi_files):
        p = pt.load_performance_midi(Path(file))
        note_array = p.note_array()
        note_sequence = np.array(list(zip(note_array['pitch'], note_array['onset_sec'], note_array['duration_sec'], note_array['velocity'])))

        key_signature_changes = processor_key_sig.process(note_sequence)

        length = note_sequence[-1][1] + note_sequence[-1][2]

        last_onset, last_key = key_signature_changes[0]
        durations = {}
        for onset, key in key_signature_changes[1:]:
            if last_key not in durations.keys():
                durations[last_key] = 0
            durations[last_key] += onset - last_onset
            last_onset = onset
            last_key = key
        if last_key not in durations.keys():
            durations[last_key] = 0
        durations[last_key] += length - last_onset

        best_duration, best_key = 0, 0
        for key, duration in durations.items():
            if duration > best_duration:
                best_duration = duration
                best_key = key

        print("Key signature changes: " + str(idx + 1) + "/" + str(len(midi_files)))
        print(key_signature_changes)
        print("Prediction: " + str(best_key))
        print(durations)

        results.append((os.path.basename(file), best_key))

    if args.outfile:
        np.savetxt(
            args.outfile,
            np.array(results),
            fmt="%s",
            delimiter=",",
            comments="//",
            header="filename,key",
        )
