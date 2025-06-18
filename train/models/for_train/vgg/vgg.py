import os
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Traverse upwards until the directory name matches the given project name
project_dir = current_dir
while os.path.basename(project_dir) != "MWA":
    project_dir = os.path.dirname(project_dir)
    if project_dir == '/':  # Breaks if we reach the root of the filesystem
        raise Exception(f"Project directory 'MWA' not found. Ensure this script is within the project directory.")

# Append the project directory to sys.path
sys.path.append(project_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from train.forcedAlignment.utils.constants import DATASETS_MAPPING, DATASET

vgg_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(vgg_dir, 'conf', 'config.yaml')

def get_fc1_input_size(model_args):
    # Load the configuration file
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract vgg_name and input_height from model_args
    vgg_name = model_args['vgg_name']
    input_height = model_args['sequence_size']
    
    # Get the fc1_input size from the config
    fc1_input_size = config['vgg_config'][DATASET][vgg_name][input_height]
    
    return fc1_input_size


def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


cfg = {
    'VGG7': [64, 'M', 128, 'M', 256],
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M'], 
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG9N': [64, 'M', 128, 'M', 256, 256, 'M'], 
    'VGG11N': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'], 
    'VGG13N': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'], 
    'VGG16N': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'VGG19N': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'], 
}


class VGG(nn.Module):
    def __init__(self, vgg_name, fc1_input_size):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, input_ids):
        # Unsqueeze the tensor to add the extra dimension
        input_ids = input_ids.unsqueeze(1)  # Shape becomes (batch_size, 1, sequence_size, 192) from (batch_size, sequence_size, 192)
        out = self.features(input_ids)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
def inizialize_vgg(model_args):
    vgg_name = model_args['vgg_name']

    # Get the fc1_input size from the config
    fc1_input_size = get_fc1_input_size(model_args)
    model = VGG(vgg_name, fc1_input_size)
    return model
    
if __name__ == '__main__':
    import torch
    model = VGG(vgg_name='VGG19N', fc1_input_size=40)
    for i in [17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47]:
        inputs = torch.randn(64,i,256)
        out = model(inputs)
        print("window size: ", i)
        print("out shape: ", out.shape)

