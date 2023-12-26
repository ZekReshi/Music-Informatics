import pytorch_lightning as pl
import os, sys

import torch

from PM2S.modules.utils import configure_callbacks, configure_optimizers, classification_report_framewise

sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn as nn

from PM2S.models.key_signature import RNNKeySignatureModel


class KeySignatureModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNKeySignatureModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, y, length = batch
        x = x.float()
        y = y.long()
        length = length.long()

        # Forward pass
        y_hat = self(x)

        # Mask out the padding part
        mask = torch.ones(y_hat.shape).to(y_hat.device)
        for i in range(y_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_hat = y_hat * mask

        # Loss
        loss = nn.NLLLoss()(y_hat, y)

        # Logging
        logs = {
            'train_loss': loss,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_size):
        # Data
        x, y, length = batch
        x = x.float()
        y = y.long()
        length = length.long()

        # Forward pass
        y_hat = self(x)

        # Mask out the padding part
        for i in range(y_hat.shape[0]):
            y_hat[i, length[i]:] = 0

        # Loss
        loss = nn.NLLLoss()(y_hat, y)

        # Metrics
        f_macro_all = 0

        for i in range(y_hat.shape[0]):
            # get sample from batch
            y_hat_i = y_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_i = y[i, :length[i]]

            # get accuracies
            (
                _, _, f_macro,
                _, _, _
            ) = classification_report_framewise(y_i, y_hat_i)

            f_macro_all += f_macro

        f_macro_all /= y_hat.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_f1': f_macro_all,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}
