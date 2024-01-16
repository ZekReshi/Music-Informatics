import warnings

warnings.filterwarnings('ignore')
import argparse
import pytorch_lightning as pl
pl.seed_everything(42)
import os

from PM2S.data.data_module import Pm2sDataModule
from PM2S.modules.key_signature import KeySignatureModule
from PM2S.modules.time_signature import TimeSignatureModule


## -------------------------
## DEBUGGING BLOCK
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.autograd.set_detect_anomaly(True)
## END DEBUGGING BLOCK
## -------------------------


def train(args):
    # Data
    data_module = Pm2sDataModule(args, feature=args.feature, full_train=args.full_train)

    # Model
    if args.feature == 'tempo':
        model = TempoModule()
    elif args.feature == 'key_signature':
        model = KeySignatureModule()
    elif args.feature == 'time_signature':
        model = TimeSignatureModule()
    else:
        raise ValueError('Invalid feature type.')

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(args.workspace, 'mlruns'),
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True
    )

    # Train
    trainer.fit(model, data_module)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('--workspace', type=str, help='Workspace directory.')
    parser.add_argument('--feature', type=str, help='Feature type.')
    parser.add_argument('--full_train', action='store_true', help='Training with the whole dataset or not (only the training set).')

    args = parser.parse_args()

    train(args)
