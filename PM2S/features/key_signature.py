import torch

from PM2S.features._processor import MIDIProcessor
from PM2S.models.key_signature import RNNKeySignatureModel
from PM2S.io.midi_read import read_note_sequence
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

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
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
    # Get one MIDI recording from the A_MAPS dataset
    midi_recording = '../../datasets/A-MAPS_1.1/MAPS_MUS-bk_xmas1_ENSTDkCl.mid'

    # Create time and key processors
    processor_key_sig = RNNKeySignatureProcessor()

    # Prediction
    key_signature_changes = processor_key_sig.process(midi_recording)

    print("\nKey signature changes:")
    print(key_signature_changes)
