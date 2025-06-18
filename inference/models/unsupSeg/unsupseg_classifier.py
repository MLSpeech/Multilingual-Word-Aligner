import argparse
from argparse import Namespace
import torch
import torchaudio
from .next_frame_classifier import NextFrameClassifier
import os
import matplotlib.pyplot as plt
import numpy as np



def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device

def list_wav_files(directory):
    wav_files = []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            full_path = os.path.join(directory, file)
            wav_files.append(full_path)
    return wav_files


def get_unsupseg_embeddings(ckpt, input_wav_file, device):
    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))
    model = NextFrameClassifier(hp).to(device)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)
        
    # load data
    audio, sr = torchaudio.load(input_wav_file)
    assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
    audio = audio.to(device)

    # run inference
    preds, cnn_represenataion = model(audio)  # get scores # change: get cnn represenatation from the model
    return cnn_represenataion[0]
        #save_model_train_cnn_presentation(wav, output_dir, cnn_represenataion)

    
    