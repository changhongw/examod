# -*- coding: utf-8 -*-

import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wav
import scipy.signal
import h5py
import json
import librosa
import multiprocessing
import argparse
from fnmatch import fnmatch
import pandas as pd

from scipy.io import wavfile
from scipy.io.wavfile import write

# find all .wav and .csv files in subdirectories
base_dir = './CBFdataset-iso-PMTs/audio/'
target = "*.wav"
wav_files = []   # create list
for path, subdirs, files in os.walk(base_dir):
    for name in files:
        if fnmatch(name, target):
            wav_files.append(os.path.join(path, name))   # add an element to the list
print('Number of audio files:', format(len(wav_files)))
    
csv_file = './CBFdataset-iso-PMTs/annotation/CBFdataset_iso_metadata.csv'
anno = pd.read_csv(csv_file)

for file in anno['file_name'].unique():
    file_anno = anno.loc[anno['file_name'] == anno['file_name'].unique()[4]].reset_index().drop(columns=['index'])
    file_dir = os.path.join(base_dir, file.split('_')[0][:-1])
    sr, audio = wavfile.read(os.path.join(file_dir, file + '.wav'))
    for k in range(len(file_anno)):
        start = file_anno['start'][k]
        end = file_anno['end'][k]
        PT_audio = audio[int(start*sr):int(end*sr)]
        write(os.path.join( file_dir, file + '_' +str(k) + '.wav'), sr, PT_audio)