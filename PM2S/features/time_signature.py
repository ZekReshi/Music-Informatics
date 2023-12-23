import torch
import argparse

from PM2S.features._processor import MIDIProcessor
from PM2S.models.time_signature import RNNTimeSignatureModel
from PM2S.constants import tsIndex2Nume


class RNNTimeSignatureProcessor(MIDIProcessor):

    def __init__(self, model_state_dict_path='_model_state_dicts/time_signature/RNNTimeSignatureModel.pth', **kwargs):
        super().__init__(model_state_dict_path, **kwargs)

    def load(self, state_dict_path):
        if state_dict_path:
            self._model = RNNTimeSignatureModel()
            self._model.load_state_dict(torch.load(state_dict_path))
        else:
            self._model = RNNTimeSignatureModel()

    def process(self, note_seq, **kwargs):
        # Read MIDI file into note sequence
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        tn_probs = self._model(x)

        # Post-processing
        tn_idx = tn_probs[0].topk(1, dim=0)[1].squeeze(0).cpu().detach().numpy() # (seq_len,)

        onsets = note_seq[:, 1]
        time_signature_changes = self.pps(tn_idx, onsets)

        return time_signature_changes

    def pps(self, tn_idx, onsets):
        ts_prev = '0/0'
        ts_changes = []
        for i in range(len(tn_idx)):
            ts_cur = '{:d}'.format(tsIndex2Nume[tn_idx[i]])
            if i == 0 or ts_cur != ts_prev:
                onset_cur = onsets[i]
                ts_changes.append((onset_cur, ts_cur))
                ts_prev = ts_cur
        return ts_changes


if __name__ == '__main__':
    import glob
    import os
    import numpy as np
    import partitura as pt
    from pathlib import Path
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Predict time signatures.')

    parser.add_argument('--datadir', type=str, help='Dataset directory.')
    parser.add_argument('--modeldir', type=str, help='Model directory.')
    parser.add_argument('--outfile', type=str, help='Output file.')

    args = parser.parse_args()

    midi_files = glob.glob(os.path.join(args.datadir, "*.mid"))
    midi_files.sort()

    # Create time and key processors
    processor_time_sig = RNNTimeSignatureProcessor(os.path.join(args.modeldir, '_model_state_dicts', 'time_signature', 'RNNTimeSignatureModel.pth'))

    results = []
    # Prediction
    for idx, file in enumerate(midi_files):
        p = pt.load_performance_midi(Path(file))
        note_array = p.note_array()
        note_sequence = np.array(list(zip(note_array['pitch'], note_array['onset_sec'], note_array['duration_sec'], note_array['velocity'])))

        time_signature_changes = processor_time_sig.process(note_sequence)

        length = note_sequence[-1][1] + note_sequence[-1][2]

        last_onset, last_ts_num = time_signature_changes[0]
        durations = {}
        for onset, ts_num in time_signature_changes[1:]:
            if not last_ts_num in durations.keys():
                durations[last_ts_num] = 0
            durations[last_ts_num] += onset - last_onset
            last_onset = onset
            last_ts_num = ts_num
        if not last_ts_num in durations.keys():
            durations[last_ts_num] = 0
        durations[last_ts_num] += length - last_onset

        best_duration, best_ts_num = 0, 0
        for ts_num, duration in durations.items():
            if duration > best_duration:
                best_duration = duration
                best_ts_num = ts_num

        print("Time signature changes: " + str(idx + 1) + "/" + str(len(midi_files)))
        print(time_signature_changes)
        print("Prediction: " + str(best_ts_num))
        print(durations)

        results.append((os.path.basename(file), best_ts_num, 120))

    np.savetxt(
        args.outfile,
        np.array(results),
        fmt="%s",
        delimiter=",",
        comments="//",
        header="filename,ts_num,tempo(bpm)",
    )
