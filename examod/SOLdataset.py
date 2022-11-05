import os
import gin
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from typing import Optional
import torch
from examod.utils import load_as_tensor

gin.enter_interactive_mode()

@gin.configurable
class SOLdataset(Dataset):
    def __init__(self,
                 data_dir='../datasets/SOL-0.9HQ-PMT',
                 csv='../datasets/SOL-0.9HQ-PMT/SOL-0.9HQ-PMT_meta.csv',
                 subset='training',
                 feature='scat1d_s1s2',
                 out_dir_to_skip=None,  # resuming mechanism
                 c = 1e-1,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.csv_dir = csv
        self.subset = subset
        self.c = c

        if out_dir_to_skip is None:
            self.out_dir_to_skip = None
        else:
            self.out_dir_to_skip = os.path.join(out_dir_to_skip, subset)

        if feature != '':
            self.feature = feature.split('_')[0]
            self.feature_spec = feature.split('_')[1]
        else: 
            self.feature = feature

        feature_dir = os.path.join(data_dir, self.feature)
        self.feature_dir = os.path.join(feature_dir, subset)

        df = pd.read_csv(os.path.join(self.csv_dir))
        
        self.df = df.loc[df['subset'] == subset]
        self.df = self.df.reset_index(drop = True)

        if (self.out_dir_to_skip is not None
                and os.path.isdir(self.out_dir_to_skip)):
            self.out_names_done = os.listdir(self.out_dir_to_skip)
        else:
            self.out_names_done = None
                
    def build_fname(self, df_item, ext='.npy'):
        filename = df_item['file_name'].split('.')[0]
        
        s1 = f'{filename}_S1{ext}'
        s2 = f'{filename}_S1S2{ext}'
        return s1, s2

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        label = item['label'] 
        fname = os.path.join(item['instrument family'], item['instrument'], 
                            item['technique original'], item['file_name'])   
            
        if self.feature=='scat1d':
            # load extracted feature for classification
            stats_dir = os.path.join(self.data_dir, self.feature)

            s1_fname, s2_fname = self.build_fname(item, '.npy')
            S1x = np.load(os.path.join(self.feature_dir, s1_fname))
            S1S2x = np.load(os.path.join(self.feature_dir, s2_fname))

            c = torch.tensor([self.c])

            # check feature_spec
            if self.feature_spec == 's1s2':
                mu = load_as_tensor(os.path.join(stats_dir, 'stats','mu_S1S2.npy'))
                Sx = torch.log1p(S1S2x / (c[:, None] * mu[:, None] + 1e-8))
                return Sx, label
            else:
                mu = load_as_tensor(os.path.join(stats_dir, 'stats','mu_S1.npy'))
                Sx = torch.log1p(S1x / (c[:, None] * mu[:, None] + 1e-8))
                return Sx, label

        else:  
            # feature='' is to do feature extraction
            audio_full, _ = torchaudio.load(os.path.join(self.data_dir, fname))
            audio = torch.mean(audio_full, 0)
        
            return audio, label, os.path.splitext(fname)[0]

    def __len__(self):
        return len(self.df)

@gin.configurable
class SOLdatasetModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = '../datasets/SOL-0.9HQ-PMT',
                 batch_size: int = 32,
                 feature='scat1d_s1s2',
                 out_dir_to_skip=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.feature = feature
        self.out_dir_to_skip = out_dir_to_skip

    def setup(self, stage: Optional[str] = None):
        self.train_ds = SOLdataset(self.data_dir, subset='training',
                                      feature=self.feature,
                                      out_dir_to_skip=self.out_dir_to_skip)
        self.val_ds = SOLdataset(self.data_dir, subset='validation',
                                    feature=self.feature,
                                    out_dir_to_skip=self.out_dir_to_skip)
        self.test_ds = SOLdataset(self.data_dir, subset='test',
                                     feature=self.feature,
                                     out_dir_to_skip=self.out_dir_to_skip)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=0)