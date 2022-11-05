import numpy as np, torch
import os

def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)

def load_as_tensor(path):
    return torch.tensor(np.load(path))

def normalize_audio(audio: np.ndarray, eps: float = 1e-10):
    return audio / (audio.std() + eps)