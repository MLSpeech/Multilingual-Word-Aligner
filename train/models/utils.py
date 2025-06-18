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


from train.forcedAlignment.utils.constants import DEVICE
import json
import torch
from train.models.for_train.vgg.vgg import inizialize_vgg
from train.models.for_train.transformer import initialize_transformer
from train.models.for_train.conformer.conformer import initialize_conformer



def calculate_trainable(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"model size: {size_all_mb:.2f} MB") 

    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"model trainable params: {num_trainable_params}")
    

def initialize_model(model='vgg', model_args={'vgg_name':'VGG11', 'sequence_size': 13}):
    if model == 'vgg':
        model = inizialize_vgg(model_args)
        calculate_trainable(model)
    elif model == 'transformer':
        model =  initialize_transformer(model_args)
        calculate_trainable(model)
    elif model == 'conformer':
        model = initialize_conformer(model_args)
        calculate_trainable(model)
    else:
        model = None 
    return model

def load_model(model_dir, device=DEVICE):
    # Define paths for the weights and configuration files
    model_weights_path = os.path.join(model_dir, 'pytorch_model.bin')
    config_path = os.path.join(model_dir, 'config.json')
    
    # Load the configuration JSON into a dictionary
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    model = initialize_model(model_config['model_type'], model_config)
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    
    print("Model and configuration loaded successfully")

    return model, model_config

