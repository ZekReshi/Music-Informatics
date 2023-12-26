
import torch.nn as nn

from PM2S.constants import ibiVocab
from PM2S.models.utils import get_in_features, encode_note_sequence
from PM2S.models.blocks import ConvBlock, GRUBlock, LinearOutput


class RNNTempoModel(nn.Module):

    def __init__(self, hidden_size=512):
        super().__init__()

        in_features = get_in_features()

        self.convs = ConvBlock(in_features=in_features)

        self.gru_tempo = GRUBlock(in_features=hidden_size)

        self.out_tempo = LinearOutput(in_features=hidden_size, out_features=ibiVocab, activation_type='softmax')

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)

        x = self.convs(x)  # (batch_size, seq_len, hidden_size)

        x_gru_tempo = self.gru_tempo(x)  # (batch_size, seq_len, hidden_size)

        y_tempo = self.out_tempo(x_gru_tempo)  # (batch_size, seq_len, ibiVocab)

        # squeeze and transpose
        y_tempo = y_tempo.transpose(1, 2)  # (batch_size, ibiVocab, seq_len)

        return y_tempo


