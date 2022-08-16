
""" Load CBFdataset test data
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import h5py
from time import time
import re
import warnings
warnings.filterwarnings('ignore')

class CBFdataset(Dataset):
    def __init__(self, split_txt):
        # Get directory listing from path
        files = pd.read_csv(split_txt, header=None)[0]
        # Iterate through the listing and create a list of tuples (filename, label)
        self.items = [(f,f.split("/")[-1].split("_")[1]) for f in files]
        self.length = len(self.items)
    def __getitem__(self, index):
        filename, _ = self.items[index]
        match = re.match(r"([a-z]+)([0-9]+)", filename.split('_')[-1], re.I)
        label = match.groups()[0]
        data = h5py.File(filename, 'r')
        featureTensor = np.array(data['data'])[0]
        # label = np.array(data['label'])[0][0]
        save_name = "_".join(filename.split('.')[0].split('\\')[-1].split('_')[1:])
        return (featureTensor, label, save_name)   
    def __len__(self):
        return self.length
    
device = torch.device("cpu"); # device="cuda:1"
bs=1

split = 0
PATH_TO_split_txt = './preprocessed_data/'

test_mnist = CBFdataset(PATH_TO_split_txt + 'AlexNet_player_' + str(split) + '_test.txt')
test_loader  = torch.utils.data.DataLoader(test_mnist, batch_size = bs, shuffle = True)
    

"""Layer-wise Relevance Propagation

This script uses a pre-trained VGG network from PyTorch's Model Zoo
to perform layer-wise relevance propagation.

"""
import time
import torch
import torchvision
import yaml

from src.visualize import plot_relevance_scores
from src.data import get_data_loader
from src.lrp import LRPModel


def run_lrp(config: dict, test_loader: object, model: object):
    """Test function that plots heatmaps for images placed in the input folder.

    Images have to be placed in their corresponding class folders.

    Args:
        config: Dictionary holding configuration parameters.

    Returns:
        None

    """
    if config["device"] == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    data_loader = test_loader
    
    model.to(device, dtype=torch.float64)

    lrp_model = LRPModel(model=model)

    for i, (x, y, z) in enumerate(data_loader):
        x = x.to(device, dtype=torch.float64)

        t0 = time.time()
        r = lrp_model.forward(x[:,:,-224:,-224:])
        print("{time:.2f} FPS".format(time=(1.0 / (time.time()-t0))))
        save_name = z[0]

        plot_relevance_scores(x=x[:,:,-224:,-224:], r=r, name=save_name, config=config)
  
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            

if __name__ == "__main__":

    with open("config.yml", "r") as stream:
        config = yaml.safe_load(stream)

    # experiment with pretrained models
    model = torchvision.models.vgg16(pretrained=True)
    # model = torchvision.models.alexnet(pretrained=True)
    # model = torchvision.models.resnet(pretrained=True)
    
    # initialize model
    set_parameter_requires_grad(model, feature_extract=True)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    num_in = model.classifier[6].in_features
    num_classes = 10
    model.classifier[6] = nn.Linear(num_in, num_classes)
    input_size = 224
    
    model.load_state_dict(torch.load("vgg16-finetuned.pth", map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load("alexnet-finetuned.pth", map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load("resnet-finetuned.pth", map_location=torch.device('cpu')))

    run_lrp(config=config, test_loader=test_loader, model=model)
    
    
