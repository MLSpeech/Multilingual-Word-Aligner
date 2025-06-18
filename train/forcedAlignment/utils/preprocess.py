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

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable
from train.forcedAlignment.utils.constants import models_configurations, LABELS_DIR, DATASETS_MAPPING, DATASET, LABELS_DIR_VAL, BUCKEYE
import pickle
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
import math

@dataclass
class PreModel:
    files_folder: str
    file_preprocess_func: Optional[Callable[[str],list]] = None  


def prepare_mms_confidents(mms_file: str, length_time_wav, files_folder, **kwargs):
    mms_file = os.path.join(files_folder,f'{mms_file}_mms.txt')
    confidents = []
    end_times = []
    transcript_started = False

    # Read the file line by line
    with open(mms_file, 'r') as file:
        for line in file:
                line = line.strip()
                
                if not transcript_started:
                    if line.startswith("Transcript:"):
                        transcript_started = True
                    continue
                
                if line.startswith("Words spans:"):
                    continue
                    #break
                
                parts = line.split()
                if len(parts) == 5:
                    end_time, score = float(parts[3][:-1]), float(parts[4])
                    confidents.append(score)
                    end_times.append(end_time)

    end_times_divide = [int(round(end_time_i *2)) for end_time_i in end_times]
    length_confidents = length_time_wav
    mms_confidents = np.zeros(length_confidents).astype(float)

    for item1, item2 in zip(end_times_divide, confidents):
        mms_confidents[(item1)-1] = item2
        mms_confidents[(item1)] = item2

    expanded_mms_confidents = np.repeat(mms_confidents[:, np.newaxis], DATASETS_MAPPING[DATASET]['mms_repeat'], axis=1)

    return expanded_mms_confidents

def prepare_clasters(file, folder, **kwargs):
    file = f'{file}_tzvia.pickle'
    with open(os.path.join(folder,file), 'rb') as handle:
        dsegknn_results = pickle.load(handle)
        results = dsegknn_results['distances']
    
    # if data is duplicate at the end - cut until the window 
    if 'model_window' in kwargs:
        results = results[:-kwargs['model_window']+1]
    
    #tzvia results are for 20ms windows, so we sould repeat to gather 10ms
    results = np.repeat(results, 2, axis=0)
    expanded_results = np.repeat(results, DATASETS_MAPPING[DATASET]['DsegkNN_repeat'], axis=1)
    
        
    return expanded_results


def prepare_cnn_representation(file, folder, **kwargs):
    file = f'{file}_UnsupSeg_model_train_cnn.pkl'
    with open(os.path.join(folder, file), 'rb') as handle:
        cnn_representation = pickle.load(handle)
    
    # Check if the representation is not a numpy array, and convert if necessary
    if not isinstance(cnn_representation, np.ndarray):
        cnn_representation = np.array(cnn_representation)

    return cnn_representation


def get_num_windows(file_path, window_size_ms=10):
    try:
        # Load the audio file using torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Calculate the duration in milliseconds
        duration_ms = waveform.size(1) / sample_rate * 1000  # size(1) is the number of samples in the audio
        
        # Calculate the number of windows
        num_windows = math.ceil(duration_ms / window_size_ms) + 1  #Example: WAV file with a length of 2.480 seconds will yield 248 frames. Adding 1 frame for the case where the label is precisely at 2.480 seconds, ensuring it is not out of bounds.
        
        return num_windows
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def prepare_labels(labels_dir, word_file: str):
    label_file = os.path.join(labels_dir, word_file)
    time_labels = []
    # Read the file line by line
    with open(label_file, 'r') as file:
        for line in file:
            # Split each line into parts (assuming whitespace separates them)
            parts = line.split()
            if len(parts) >= 3:
                # The end time is the second element converted to an integer
                end_time = int(parts[1])
                # Append the end time to the list
                time_labels.append(end_time)

    time_labels_divide = [int(time_label // 160) for time_label in time_labels]
    wav_file = os.path.splitext(label_file)[0] + '.wav'
    length_time_labels = get_num_windows(wav_file, window_size_ms=10)
    labels = np.zeros(length_time_labels).astype(int)
    for i in time_labels_divide:
        labels[i] = 1

    return labels, length_time_labels

def get_labels_in_sec(labels_dir, word_file):
    label_file = os.path.join(labels_dir, word_file)
    time_labels = []
    # Read the file line by line
    with open(label_file, 'r') as file:
        for line in file:
            # Split each line into parts (assuming whitespace separates them)
            parts = line.split()
            if len(parts) >= 3:
                # The end time is the second element converted to an integer
                end_time = float(parts[1])
                # Append the end time to the list
                time_labels.append(end_time)

    time_labels_divide = [time_label / 16000 for time_label in time_labels]
    return time_labels_divide




def get_all_model_files_names(labels_dir=LABELS_DIR, specific_file=None):
    """
    preparing the datasets files
    """
    # Initialize lists to store the file names
    labels_files = []
    posix_label = '.word' if (BUCKEYE in labels_dir) else '.wrd'
    # Iterate through the files in the directory
    for filename in os.listdir(labels_dir):
        if filename.endswith(posix_label) and (specific_file is None or specific_file == filename):
            # If it's a .wrd file, add it to the speech files list
            labels_files.append(filename)

    # Sort the lists to ensure they match up correctly
    labels_files.sort()
    return labels_files

def contrast_normalization(arr):
    # Get the min and max values from the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Avoid division by zero if all values are the same
    if arr_min == arr_max:
        return np.zeros_like(arr)  # Return an array of zeros if no contrast can be applied

    # Normalize the array to the range [0, 1]
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    
    return normalized_arr

def pad_and_batch_transformer(final_embedding, labels, sequence_size=200):
    """
    Pad the final embeddings and labels to be divisible by batch size and split into batches.
    
    Args:
        final_embedding (numpy.ndarray): The final embedding matrix (frames x embedding_dim).
        labels (numpy.ndarray): The labels corresponding to the embeddings.
        sequence_size (int): The batch size to pad and split to. Default is 200.
        
    Returns:
        list: A list of batches for embeddings.
        list: A list of batches for labels.
    """
    frames_length = final_embedding.shape[0]
    embedding_length = final_embedding.shape[1]
    
    # Calculate padding length
    padding_length = (frames_length // sequence_size + 1) * sequence_size - frames_length
    
    # Pad embeddings
    padding_embedding = np.zeros((padding_length, embedding_length))
    final_embedding = np.vstack((final_embedding, padding_embedding))
    
    # Pad labels
    labels = np.concatenate([labels, np.zeros(padding_length)])
    
    # Split into batches
    final_embedding_batches = np.split(final_embedding, final_embedding.shape[0] // sequence_size)
    label_batches = np.split(labels, labels.shape[0] // sequence_size)
    
    return final_embedding_batches, label_batches

def pad_and_batch_transformer_and_conformer(final_embedding, labels, sequence_size=200):
    """
    Pad the final embeddings and labels to be divisible by batch size and split into batches.
    
    Args:
        final_embedding (numpy.ndarray): The final embedding matrix (frames x embedding_dim).
        labels (numpy.ndarray): The labels corresponding to the embeddings.
        sequence_size (int): The batch size to pad and split to. Default is 200.
        
    Returns:
        numpy.ndarray: A numpy array of batches for embeddings.
        numpy.ndarray: A numpy array of batches for labels.
        numpy.ndarray: A numpy array of batches for masks.
    """
    frames_length = final_embedding.shape[0]
    embedding_dim = final_embedding.shape[1]  # size of each embedding vector

    final_embedding_batches = []
    label_batches = []
    mask_batches = []

    if frames_length < sequence_size:
        # Calculate the amount of padding needed
        padding_length = sequence_size - frames_length

        # Create a padding array filled with zeros
        label_padding = np.zeros((padding_length,), dtype=labels.dtype)
        labels_padded = np.concatenate([labels, label_padding])
        label_batches.append(labels_padded)
    
        # Create a padding array of shape (padding_length, embedding_dim) filled with -inf
        padding = np.full((padding_length, embedding_dim), 0.0) #-np.inf
        final_embedding_padded = np.vstack([final_embedding, padding])
        final_embedding_batches.append(final_embedding_padded)

        mask = np.ones(sequence_size, dtype=bool)
        mask[frames_length:] = False
        mask_batches.append(mask)
  
    else:
        for start in range(0, frames_length - sequence_size + 1, sequence_size): 
            sub_embedding = final_embedding[start:start + sequence_size]
            final_embedding_batches.append(sub_embedding)
            sub_label = labels[start:start + sequence_size]
            label_batches.append(sub_label)

            # Create the mask: 
            mask = np.ones(sequence_size, dtype=bool)
            mask_batches.append(mask)

        if frames_length % sequence_size != 0:
            start = frames_length - sequence_size
            sub_embedding = final_embedding[start:]
            final_embedding_batches.append(sub_embedding)
            sub_label = labels[start:]
            label_batches.append(sub_label)

            mask = np.ones(sequence_size, dtype=bool)
            num_zeros = sequence_size - (frames_length % sequence_size)
            mask[:num_zeros] = False
            mask_batches.append(mask)

    # Convert batches to numpy arrays before returning
    final_embedding_batches = np.array(final_embedding_batches)
    label_batches = np.array(label_batches)
    mask_batches = np.array(mask_batches)

    return final_embedding_batches, label_batches, mask_batches


def pad_and_batch_cnn(final_embedding, labels, sequence_size=11, labels_size=1):
    """
    Pad the final embeddings and labels to resemble how CNN works with sequence-based inputs.
    Each frame will be padded with context on both sides, and split into batches.

    Args:
        final_embedding (numpy.ndarray): The final embedding matrix (frames x embedding_dim).
        labels (numpy.ndarray): The labels corresponding to the embeddings.
        sequence_size (int): Size of the context window (sequence). Default is 11.
        
    Returns:
        numpy.ndarray: A matrix of sliding windows for embeddings.
        numpy.ndarray: A matrix of sliding windows for labels.
    """
    frames_length = final_embedding.shape[0]
    embedding_length = final_embedding.shape[1]
    
    # Calculate padding: half sequence size on left and right
    pad_size = sequence_size // 2
    
    # Pad embeddings on both sides (left and right)
    padding_embedding = np.zeros(((2 * pad_size), embedding_length))
    padded_embedding = np.vstack((padding_embedding[:pad_size], final_embedding, padding_embedding[-pad_size:]))
    
    padded_labels = labels
    # Use as_strided to create sliding windows for embeddings
    embedding_windows = np.lib.stride_tricks.as_strided(
        padded_embedding,
        shape=(frames_length, sequence_size, embedding_length),
        strides=(padded_embedding.strides[0], padded_embedding.strides[0], padded_embedding.strides[1])
    )
    
    # Use as_strided to create sliding windows for labels
    label_windows = np.lib.stride_tricks.as_strided(
        padded_labels,
        shape=(frames_length, labels_size),
        strides=(padded_labels.strides[0], padded_labels.strides[0])
    )
    mask = np.ones((frames_length, 1), dtype=bool)
    
    return embedding_windows, label_windows, mask


def prepare_dataset(labels_dir=LABELS_DIR, mode='train', model_arguments =None, specific_file=None):
    model = model_arguments['model_type']
    
    if mode == 'train':
        configuration_folder_name = 'train_folder'
    elif mode == 'val':
        configuration_folder_name = 'val_folder'
    elif mode == 'test':
        configuration_folder_name = 'test_folder'
    else:
        raise Exception("enter a valid mode")
    
    mms = PreModel(
            files_folder=models_configurations['mms'][configuration_folder_name],
            file_preprocess_func=prepare_mms_confidents
        )
    # dsegknn = PreModel(
    #         files_folder=models_configurations['dsegknn'][configuration_folder_name],
    #         file_preprocess_func=prepare_clasters
    #     )
    UnsupSeg = PreModel(
            files_folder=models_configurations['UnsupSeg'][configuration_folder_name],
            file_preprocess_func=prepare_cnn_representation
        )
    
    all_embeddings = []
    all_labels = []
    all_masks = []

    labels_files = get_all_model_files_names(labels_dir=labels_dir, specific_file=specific_file)
    for labels_file in labels_files:
        try:
            file_name = Path(labels_file).stem
            labels, length_time_labels = prepare_labels(labels_dir, labels_file)   
            mms_confidents = mms.file_preprocess_func(file_name, length_time_labels, mms.files_folder, **models_configurations['mms'])
            UnsupSeg_cnn_represenataion = UnsupSeg.file_preprocess_func(file_name, UnsupSeg.files_folder, **models_configurations['UnsupSeg'])
            
            min_embedd_shape = min(mms_confidents.shape[0], labels.shape[0], UnsupSeg_cnn_represenataion.shape[0]) 
            # Cutting all vectors to minimal sized vector
            mms_confidents = mms_confidents[:min_embedd_shape]
            labels = labels[:min_embedd_shape]
            UnsupSeg_cnn_represenataion = UnsupSeg_cnn_represenataion[:min_embedd_shape]
            
            # Norm UnSupSeg
            norm_UnsupSeg_cnn_represenataion = contrast_normalization(UnsupSeg_cnn_represenataion)
            
            final_embedding = np.hstack((norm_UnsupSeg_cnn_represenataion, mms_confidents))
            
            if model=='transformer' or model == 'conformer':
                # split to batches and paddinf for transformer.
                final_embedding_batches, label_batches, mask_batches = pad_and_batch_transformer_and_conformer(final_embedding, labels, sequence_size=model_arguments['sequence_size'])
            elif model=='vgg':
                final_embedding_batches, label_batches, mask_batches = pad_and_batch_cnn(final_embedding, labels, sequence_size=model_arguments['sequence_size'], labels_size=model_arguments['labels_size'])
            else:
                raise ValueError(f"Unsupported model type: {model}. Please use 'transformer' or 'VGG'.")
            
            all_embeddings.extend(final_embedding_batches)
            all_labels.extend(label_batches)
            all_masks.extend(mask_batches)
            
        except Exception as e:
            print(f"file is corrupted {labels_file} - {e}")
        
    return all_embeddings, all_labels, all_masks


class EmbeddedDataset(Dataset):
    def __init__(self, embedded_sentences_list, labels_list, mask_list):
        self.embedded_sentences_list = embedded_sentences_list
        self.labels_list = labels_list
        self.mask_list = mask_list

    def __len__(self):
        return len(self.embedded_sentences_list)

    def __getitem__(self, idx):
        embedded_sentences = self.embedded_sentences_list[idx]
        labels = self.labels_list[idx]
        mask = self.mask_list[idx]

        # Convert to tensors
        padded_embeddings = torch.tensor(embedded_sentences, dtype=torch.float32)
        padded_labels = torch.tensor(labels, dtype=torch.float32)  # Wrap labels in a list to make it a tensor
        padded_mask = torch.tensor(mask, dtype=torch.float32)

        return padded_embeddings, padded_labels, padded_mask



if __name__ == '__main__':
    
    model_arguments = {'model_type':'conformer','sequence_size':300, 'labels_per_input':300,'number_attention_heads':12, 
                                    'conformer_blocks': 16, 'karnel_size': 7}
    file_name = 'dr5_fcal1_sx53.wrd'  #file name for article pictures
    all_embeddings, all_labels, all_masks = prepare_dataset(labels_dir=LABELS_DIR_VAL, mode='val', model_arguments=model_arguments, specific_file=file_name)