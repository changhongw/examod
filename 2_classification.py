# run configs ################################################################
# random seed
SEED = 0
# `None` for training
TEST_PATH = None
# number of train epochs
MAX_EPOCHS = 500
# batch size
BATCH_SIZE = 32
# wandb
LOG = True
# path to config file
GIN_CONFIG_FILE = 'scripts/gin/config.gin'

# code #######################################################################
import gin
import pytorch_lightning as pl
pl.seed_everything(SEED)
from pytorch_lightning.callbacks import (ModelCheckpoint, TQDMProgressBar, EarlyStopping)
from pytorch_lightning.loggers import WandbLogger
from pprint import pprint
import torch
import pandas as pd
import os

from examod.cnn import SOLdatasetClassifier
from examod.SOLdataset import SOLdatasetModule, SOLdataset

import warnings
warnings.filterwarnings("ignore")

gin.enter_interactive_mode()

def main():
    gin.parse_config_file(GIN_CONFIG_FILE)

    # build model & dataset ##################################################
    dataset = SOLdatasetModule(batch_size=BATCH_SIZE)
    feature = dataset.feature

    progbar_callback = TQDMProgressBar(refresh_rate=50)
    wandb_logger = WandbLogger(project=feature) if LOG else None
    checkpoint_kw = dict(
        filename= feature + '-{epoch}-val_acc{val/acc:.3f}-val_loss{val/loss:.3f}',
        monitor='val/acc',
        mode='max',
        save_top_k=1,
        save_last=True,
    )
    checkpoint_cb = ModelCheckpoint(**checkpoint_kw)
    # number of samples per epoch: use all training data to train
    df = pd.read_csv(SOLdataset().csv_dir)
    EPOCH_SIZE = len(df[df['subset']=='training'])
    n_batches_train = EPOCH_SIZE // BATCH_SIZE

    # train or load to test ##################################################
    early_stopping = EarlyStopping(monitor = 'val/acc', patience=20, mode='max')

    trainer = pl.Trainer(# gpus=-1,
                         max_epochs=MAX_EPOCHS,
                         callbacks=[progbar_callback, checkpoint_cb, early_stopping],
                        #  fast_dev_run=True,
                         limit_train_batches=n_batches_train,
                         logger=wandb_logger)

    if TEST_PATH is None:
        model = SOLdatasetClassifier(n_batches_train=n_batches_train)  
        trainer.fit(model, dataset)
        torch.save(model, os.getcwd() + '/results/cnn_trained_' + feature.split('_')[-1] + '.pt')
    else:
        model = SOLdatasetClassifier.load_from_checkpoint(TEST_PATH)
    
    # test ###################################################################
    x = trainer.test(model, dataset, verbose=False)
    results = {'acc_macro': x[0]['acc_macro'],
               'acc_classwise': [float(i) for i in x[0]['acc_classwise'].values()],
               'val_acc': x[0].get('val_acc', -1),
               'val_loss': x[0].get('val_loss', -1)}
    pprint(results)
    if TEST_PATH is not None:
        print(TEST_PATH)

if __name__ == "__main__":
    main()