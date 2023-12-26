import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn as nn

from PM2S.models.tempo import RNNTempoModel
from PM2S.modules.utils import *


class TempoModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNTempoModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_idx):
        # Data
        x, y_tempo, length = batch
        x = x.float()
        y_tempo = y_tempo.long()
        length = length.long()

        # Forward pass
        y_tempo_hat = self(x)

        # Mask out the padding part
        mask = torch.ones(y_tempo_hat.shape).to(y_tempo_hat.device)
        for i in range(y_tempo_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_tempo_hat = y_tempo_hat * mask

        # Loss
        loss = nn.NLLLoss()(y_tempo_hat, y_tempo)

        # Logging
        logs = {
            'train_loss': loss,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_idx):
        # Data
        x, y_tempo, length = batch
        x = x.float()
        y_tempo = y_tempo.long()
        length = length.long()

        # Forward pass
        y_tempo_hat = self(x)

        # Mask out the padding part
        for i in range(y_tempo_hat.shape[0]):
            y_tempo_hat[i, :, length[i]:] = 0

        # Loss
        loss = nn.NLLLoss()(y_tempo_hat, y_tempo)

        # Metrics
        accs, precs, recs, fs = 0, 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_tempo_hat_i = y_tempo_hat[i, :, :length[i]].topk(1, dim=0)[1][0]

            y_tempo_i = y_tempo[i, :length[i]]

            # filter out ignore indexes
            y_tempo_hat_i = y_tempo_hat_i[y_tempo_i != 0]
            y_tempo_i = y_tempo_i[y_tempo_i != 0]

            # get accuracy
            acc, prec, rec, f = f_measure_framewise(y_tempo_i, y_tempo_hat_i)
            
            accs += acc
            precs += prec
            recs += rec
            fs += f

        accs /= x.shape[0]
        precs /= x.shape[0]
        recs /= x.shape[0]
        fs /= x.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_f1': fs,
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}
