# run configs ################################################################
# random seed
SEED = 0
# `None`: train -- str: load model to test
TEST_PATH = None
# number of train epochs
N_EPOCHS = 50
# batch size
BATCH_SIZE = 32
# wandb
LOG = False
# path to config file
GIN_CONFIG_FILE = 'scripts/gin/config.gin'

# code #######################################################################
import gin, fire
import pytorch_lightning as pl
pl.seed_everything(SEED)
from pytorch_lightning.callbacks import (ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.loggers import WandbLogger
from pprint import pprint
import torch
import pandas as pd

from examod.cnn import SOLdatasetClassifier
from examod.SOLdataset import SOLdatasetModule, SOLdataset

import warnings
warnings.filterwarnings("ignore")

gin.enter_interactive_mode()

def run_model():
    gin.parse_config_file(GIN_CONFIG_FILE)

    # build model & dataset ##################################################
    dataset = SOLdatasetModule(batch_size=BATCH_SIZE)
    feature = dataset.feature

    progbar_callback = TQDMProgressBar(refresh_rate=50)
    wandb_logger = WandbLogger(project=feature) if LOG else None
    checkpoint_kw = dict(
        filename= feature + '-{step}-val_acc{val/acc:.3f}-val_loss{val/loss:.3f}',
        monitor='val/acc',
        mode='max',
        every_n_epochs=1,
        save_top_k=-1,
    )
    checkpoint_cb = ModelCheckpoint(**checkpoint_kw)
    # number of samples per epoch: use all training data to train
    df = pd.read_csv(SOLdataset().csv_dir)
    EPOCH_SIZE = len(df[df['subset']=='training'])
    n_batches_train = EPOCH_SIZE // BATCH_SIZE

    trainer = pl.Trainer(# gpus=-1,
                         max_epochs=N_EPOCHS,
                         callbacks=[progbar_callback, checkpoint_cb],
                        #  fast_dev_run=True,
                         limit_train_batches=n_batches_train,
                         logger=wandb_logger)

    # train / load to test
    if TEST_PATH is None:
         # instantiate CNN: __init__ of both "SOLdatasetClassifier" and "CNN1D"
        model = SOLdatasetClassifier(n_batches_train=n_batches_train)  
        trainer.fit(model, dataset)
        torch.save(model, 'results/cnn_trained_' + feature.split('_')[-1] + '.pt')
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

# python 2_classification.py
# def main():
#   fire.Fire(run_model())

if __name__ == "__main__":
    # main()
    run_model()