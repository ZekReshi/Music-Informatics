import os
import sys

import torch.nn as nn

from PM2S.models.time_signature import RNNTimeSignatureModel
from PM2S.modules.utils import *

sys.path.insert(0, os.path.join(sys.path[0], '..'))


class TimeSignatureModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNTimeSignatureModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, y_tn, length = batch
        x = x.float()
        y_tn = y_tn.long()
        length = length.long()

        # Forward pass
        y_tn_hat = self(x)

        # Mask out the padding part
        pad_mask = torch.ones((y_tn_hat.shape[0], y_tn_hat.shape[2])).to(y_tn_hat.device)
        for i in range(y_tn_hat.shape[0]):
            pad_mask[i, length[i]:] = 0
        y_tn_hat = y_tn_hat * pad_mask.unsqueeze(1)

        # Loss
        loss = nn.NLLLoss(ignore_index=0)(y_tn_hat, y_tn)

        # Logging
        logs = {
            'train_loss': loss,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_size):
        # Data
        x, y_tn, length = batch
        x = x.float()
        y_tn = y_tn.long()
        length = length.long()

        # Forward pass
        y_tn_hat = self(x)

        # Mask out the padding part
        for i in range(y_tn_hat.shape[0]):
            y_tn_hat[i, :, length[i]:] = 0

        # Loss
        loss = nn.NLLLoss(ignore_index=0)(y_tn_hat, y_tn)

        # Metrics
        precs_macro_tn, recs_macro_tn, fs_macro_tn = 0, 0, 0
        precs_weighted_tn, recs_weighted_tn, fs_weighted_tn = 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_tn_hat_i = y_tn_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_tn_i = y_tn[i, :length[i]]

            # filter out ignored indexes (the same as padding)
            y_tn_hat_i = y_tn_hat_i[y_tn_i != 0]
            y_tn_i = y_tn_i[y_tn_i != 0]

            # get accuracies
            (
                prec_macro_tn, rec_macro_tn, f_macro_tn,
                prec_weighted_tn, rec_weighted_tn, f_weighted_tn
            ) = classification_report_framewise(y_tn_i, y_tn_hat_i)
            
            precs_macro_tn += prec_macro_tn
            recs_macro_tn += rec_macro_tn
            fs_macro_tn += f_macro_tn
            precs_weighted_tn += prec_weighted_tn
            recs_weighted_tn += rec_weighted_tn
            fs_weighted_tn += f_weighted_tn

        precs_macro_tn /= x.shape[0]
        recs_macro_tn /= x.shape[0]
        fs_macro_tn /= x.shape[0]
        precs_weighted_tn /= x.shape[0]
        recs_weighted_tn /= x.shape[0]
        fs_weighted_tn /= x.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_f1_tn': fs_macro_tn,
            'val_f1': (fs_macro_tn) / 2,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}






