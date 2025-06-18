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
from transformers import BertModel, BertConfig, BertForTokenClassification
import torch.nn as nn
from transformers import HubertModel, HubertConfig
import torch.nn.functional as F
from train.forcedAlignment.utils.constants import DATASETS_MAPPING, DATASET

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



####################################################################################################################################################################################################
    
class BertForSequenceLabeling(nn.Module):
    def __init__(self, model_args):
        super(BertForSequenceLabeling, self).__init__()
        # Define a custom configuration for the transformer
        config = BertConfig(
            vocab_size=9192,
            hidden_size=DATASETS_MAPPING[DATASET]['transformer_size'],             
            num_hidden_layers=model_args['attention_size'],        
            num_attention_heads=model_args['attention_size'],       # Adjusted to match hidden_size (192 / 6 = 32)
            intermediate_size=768,       
            max_position_embeddings=model_args['sequence_size']
        )

        # Initialize the model
        self.model = BertModel(config)
        # Add a classification head: a linear layer that maps from 768 to 1 for binary classification
        self.classifier = nn.Linear(DATASETS_MAPPING[DATASET]['transformer_size'], 1)     
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids):
        # Get BERT outputs (last hidden state is the first output)
        outputs = self.model(inputs_embeds=input_ids)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, 192)
        
        # Apply the classifier to the CLS token representation
        logits = self.classifier(sequence_output)  # Shape: (batch_size, 1)

        return logits #, sequence_output, input_ids


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
    


def initialize_transformer(model_args):
    model = BertForSequenceLabeling(model_args)
    return model




if __name__ == '__main__':
    # Define a custom configuration for the transformer
    config = BertConfig(
        vocab_size=9192,
        hidden_size=192,             
        num_hidden_layers=8,        
        num_attention_heads=8,       # Adjusted to match hidden_size (192 / 6 = 32)
        intermediate_size=768,       
        max_position_embeddings=350
    )

    # Initialize the model
    model = BertModel(config)
    print(model)

    calculate_trainable(model)

    inputs = torch.randn(64,200,192)
    model(inputs_embeds=inputs)
