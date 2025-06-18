
import torch
import os
import sys
import time

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

import dill
from argparse import Namespace
import torch
import torchaudio
from train.models.UnSupSeg.utils import (detect_peaks, max_min_norm, replicate_first_k_frames)
from train.models.UnSupSeg.next_frame_classifier import NextFrameClassifier
from train.models.UnSupSeg.dataloader import save_score_presentation, load_tensor_from_pickle, save_model_train_cnn_presentation
from train.forcedAlignment.utils.constants import UNSUPSEG_DIR, PATH_TO_DATA_DIR, DATASET, MODEL_TRAINED_DATASET, TIMIT
import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import pickle
print(dill.__version__)




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


def unsupseg_pred(input_wav, ckpt_path=None, prominence=None):
    if ckpt_path is not None:
        ckpt = ckpt_path
    else:
        ckpt = os.path.join(project_dir, f'train/models/UnsupSeg/pretrained_models/timit_pretrained.ckpt')

    device = get_device()

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))
    # load weights and peak detection params
    model = NextFrameClassifier(hp).to(device)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)

    # Examine and process the peak_detection_params data
    peak_detection_params = {
    'prominence': 0.2,
    'width': None,
    'distance': None,
    'epoch': 4
    }
 
    if prominence is not None:
        print(f"overriding prominence with {prominence}")
        peak_detection_params["prominence"] = prominence

    # load data
    audio, sr = torchaudio.load(input_wav)
    assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
    audio = audio[0]
    audio = audio.unsqueeze(0).to(device)

    # run inference
    preds, _ = model(audio)  # get scores # change: get cnn represenatation from the model

    preds = preds[1][0]  # get scores of positive pairs
    preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
    preds = 1 - max_min_norm(preds)  # normalize scores (good for visualizations)
    preds_peaks = detect_peaks(x=preds,
                         lengths=[preds.shape[1]],
                         prominence=peak_detection_params["prominence"],
                         width=peak_detection_params["width"],
                         distance=peak_detection_params["distance"])  # run peak detection on scores
    #preds = preds[0] * 160 / sr  # transform frame indexes to seconds


    return preds, preds_peaks

        


def main(ckpt, output_dir, input_dir):
    print(f"running inferece using ckpt: {ckpt}")
    print("\n\n", 90 * "-")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wav_files_lst = list_wav_files(input_dir)

    device = get_device()

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))
    # load weights and peak detection params
    model = NextFrameClassifier(hp).to(device)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)
    
    for wav in wav_files_lst:

        # load data
        audio, sr = torchaudio.load(wav)
        assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
        audio = audio[0]
        audio = audio.unsqueeze(0).to(device)

        # run inference
        _, cnn_represenataion = model(audio)  # get scores + get cnn represenatation from the model

        #save_score_presentation(wav, output_dir, preds)
        save_model_train_cnn_presentation(wav, output_dir, cnn_represenataion)

if __name__ == "__main__":
    
    for mode in ['train','val', 'test']:
        input_dir = os.path.join(PATH_TO_DATA_DIR, f'{DATASET}/{mode}/')
        output_dir = os.path.join(project_dir, f'train/models/UnsupSeg/output_UnSupSeg_plus_cnn/{UNSUPSEG_DIR}/{mode}')

        if MODEL_TRAINED_DATASET == TIMIT:
            #ckpt = os.path.join(project_dir, f'train/models/UnsupSeg/pretrained_models/timit_pretrained.ckpt')
            ckpt = os.path.join(project_dir, f'train/models/UnsupSeg/pretrained_models/timit+_pretrained.ckpt')
        else:
            #os.path.join(project_dir, f'train/models/UnsupSeg/pretrained_models/buckeye_pretrained.ckpt')
            ckpt= os.path.join(project_dir, f'train/models/UnsupSeg/pretrained_models/buckeye+_pretrained.ckpt')

        main(ckpt, output_dir, input_dir)
