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
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from train.forcedAlignment.utils.constants import DEVICE, DATASETS_MAPPING, DATASET, TIMIT, MMS_EMISSIONS_FOLDER, LABELS_DIR, LABELS_DIR_VAL, TEST_FILES
from train.models.MMS.mms import prepare_data, get_emission
from train.forcedAlignment.utils.preprocess import get_all_model_files_names
import pickle

# Function to plot the trellis matrix
def plot_trellis(trellis, save_path):
    fig, ax = plt.subplots()
    img = ax.imshow(trellis.T, origin="lower")
    ax.set_title("Trellis Matrix")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()

    # Save the plot to the specified path without displaying it
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close(fig)  # Close the figure to release memory


def normalize_tensor(trellis):
    # Create a copy of the tensor to avoid modifying the original
    norm_rellis = trellis.clone()
    
    # Iterate over each row
    for i in range(trellis.size(0)):
        row = trellis[i]
        
        # Find valid (non-inf) values in the row, excluding both +inf and -inf
        valid_values = row[(row != float('inf')) & (row != float('-inf'))]
        
        if valid_values.numel() > 0:  # Check if there are valid values
            row_min = valid_values.min()
            row_max = valid_values.max()
            
            # Normalize values in the row
            norm_rellis[i] = (row - row_min) / (row_max - row_min)
        
        # Convert inf values to 0
        norm_rellis[i][(row == float('inf')) | (row == float('-inf'))] = 0
        
    return norm_rellis



@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0 and t > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

def get_prob_word(trellis, norm_trellis, emission, tokens, token_indices, word_index, times):
    
    time_start, time_end = times
    # Calculate the column range for the word in the trellis matrix
    num_chars_before_word = sum(len(word) for word in tokens[:word_index-1])
    num_chars_in_word = len(tokens[word_index-1])
    start_col = num_chars_before_word
    end_col = num_chars_before_word + num_chars_in_word

    token_indices_sliced = token_indices[start_col:end_col]
    sliced_emission = emission[time_start:time_end + 1, :]
    sliced_trellis = trellis[time_start:time_end + 1, start_col:end_col]

    path = backtrack(sliced_trellis.cpu().detach(), sliced_emission.cpu().detach(), token_indices_sliced)
    score_word = 0
    for p in path:
        row = p.time_index + time_start
        column = p.token_index + start_col
        value = norm_trellis[row][column]
        if torch.isnan(value):  # Check if the value is NaN
            value = torch.tensor(0.0)  # Set NaN to 0
        score_word += value.item()
    score_word = score_word / (time_end - time_start + 1)
    return score_word

# Function to generate trellis matrix
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros(num_frame, num_tokens)
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],  # Score for staying at the same token
            trellis[t, :-1] + emission[t, tokens[1:]],  # Score for changing to the next token
        )
    return trellis

def score_boundaries(norm_trellis, tokens):
    score_bound = torch.zeros((len(tokens)-1, norm_trellis.shape[0]))
    print("len(tokens)-1: ", len(tokens)-1)
    print("norm_trellis.shape[0]: ", norm_trellis.shape[0])
    num_letters = 0
    for i in range(len(tokens)-1):
        word = tokens[i]
        row = len(word) -1 + num_letters
        num_letters += len(word)
        print("word: ", word)
        print("len word: ", len(word))
        for frame in range (norm_trellis.shape[0]):
            score_frame = norm_trellis[frame][row].item() + norm_trellis[frame][row+1].item()
            score_bound[i][frame] = score_frame/2
    
    return score_bound

def get_emmissions_not_norm(labels_dir, word_file, emission_path=MMS_EMISSIONS_FOLDER):
    
    transcript_path = os.path.join(labels_dir, word_file)
    wav_path = os.path.splitext(transcript_path)[0] + '.wav'
    with_star = False  # Set to True if you want to use the 'star' option in the tokenizer
    waveform, transcript, model, tokenizer, words_len = prepare_data(wav_path, transcript_path, with_star, DEVICE)
    bundle = torchaudio.pipelines.MMS_FA
    labels = bundle.get_labels()
    tokens = transcript.split()
    dictionary = {label: idx for idx, label in enumerate(labels)}
    token_indices = []
    for word in tokens:
        word_tokens = list(word)
        token_indices.append([dictionary.get(char, dictionary['*']) for char in word_tokens])

    # If there are no words in the transcript, skip
    if words_len == 0:
        print("Empty transcript, skipping.")
        return
    
    # Get emission (log-softmax probabilities) on GPU 0
    emission = get_emission(waveform, model, DEVICE)
    emission = emission.repeat_interleave(2, dim=1)[0]
    #emission  = torch.clamp(emission, 0, 1)
    emission = emission.cpu()
    return emission


def extract_file_emissions_token(labels_dir, word_file, emission_path=MMS_EMISSIONS_FOLDER):
    
    posix_label = '.wrd' if TIMIT in labels_dir else '.word'
    pkl_file = word_file.replace(posix_label, '.pkl')
    if os.path.exists(os.path.join(MMS_EMISSIONS_FOLDER,pkl_file)):
        with open(os.path.join(emission_path,pkl_file), 'rb') as f:
            full_results = pickle.load(f)
        return full_results['tokens'], full_results['token_indices'],full_results['emissions']
    
    transcript_path = os.path.join(labels_dir, word_file)
    wav_path = os.path.splitext(transcript_path)[0] + '.wav'
    with_star = False  # Set to True if you want to use the 'star' option in the tokenizer
    waveform, transcript, model, tokenizer, words_len = prepare_data(wav_path, transcript_path, with_star, DEVICE)
    bundle = torchaudio.pipelines.MMS_FA
    labels = bundle.get_labels()
    tokens = transcript.split()
    dictionary = {label: idx for idx, label in enumerate(labels)}
    token_indices = []
    for word in tokens:
        word_tokens = list(word)
        token_indices.append([dictionary.get(char, dictionary['*']) for char in word_tokens])

    # If there are no words in the transcript, skip
    if words_len == 0:
        print("Empty transcript, skipping.")
        return
    
    # Get emission (log-softmax probabilities) on GPU 0
    emission = get_emission(waveform, model, DEVICE)
    emission = emission.repeat_interleave(2, dim=1)[0]
    emission = torch.softmax(emission, dim=-1)
    emission  = torch.clamp(emission, 0, 1)
    emission = emission.cpu()
    return tokens, token_indices, emission
    





def save_emissions_to_file(emission_path):
    
    labels_files = get_all_model_files_names(labels_dir=LABELS_DIR)
    labels_files_val = get_all_model_files_names(labels_dir=LABELS_DIR_VAL)
    labels_files_test = get_all_model_files_names(labels_dir=TEST_FILES)
    for label_file in labels_files:
        try:
            tokens, token_indices, emissions = extract_file_emissions_token(LABELS_DIR, label_file)
            emissions =  emissions.cpu()
            full_results = {'tokens':tokens,'token_indices':token_indices, 'emissions':emissions}
            posix_label = '.wrd' if TIMIT in LABELS_DIR else '.word'
            label_file = label_file.replace(posix_label, '.pkl')
            
            with open(os.path.join(emission_path,label_file), 'wb') as f:
                pickle.dump(full_results, f)
        except Exception as e:
            print(f"problem with file {label_file} - {e}")
            
    print("finished 1")
    for label_file in labels_files_val:
        try:
            tokens, token_indices, emissions = extract_file_emissions_token(LABELS_DIR_VAL, label_file)
            emissions =  emissions.cpu()
            full_results = {'tokens':tokens,'token_indices':token_indices, 'emissions':emissions}
            posix_label = '.wrd' if TIMIT in LABELS_DIR_VAL else '.word'
            label_file = label_file.replace(posix_label, '.pkl')
            
            with open(os.path.join(emission_path,label_file), 'wb') as f:
                pickle.dump(full_results, f)
        except Exception as e:
            print(f"problem with file {label_file} - {e}")      

    for label_file in labels_files_test:
            try:
                tokens, token_indices, emissions = extract_file_emissions_token(TEST_FILES, label_file)
                emissions =  emissions.cpu()
                full_results = {'tokens':tokens,'token_indices':token_indices, 'emissions':emissions}
                posix_label = '.wrd' if TIMIT in TEST_FILES else '.word'
                label_file = label_file.replace(posix_label, '.pkl')
                
                with open(os.path.join(emission_path,label_file), 'wb') as f:
                    pickle.dump(full_results, f)
            except Exception as e:
                print(f"problem with file {label_file} - {e}")     
        
    # for label_file in labels_files_val:
    #     tokens, token_indices, emissions = extract_file_emissions_token(LABELS_DIR_VAL, label_file)
    #     emissions =  emissions.cpu()
        
# def check_time():
#     labels_files = get_all_model_files_names(labels_dir=LABELS_DIR)
#     emission_statistics = []
#     for label_file in labels_files:
#         start_batch= time.time()
#         posix_label = '.wrd' if TIMIT in LABELS_DIR_VAL else '.word'
#         label_file = label_file.replace(posix_label, '.pkl')
#         if os.path.exists(os.path.join(MMS_EMISSIONS_FOLDER,label_file)):
#             posix_label = '.wrd' if TIMIT in LABELS_DIR else '.word'
#             label_file = label_file.replace(posix_label, '.pkl')
#             with open(os.path.join(MMS_EMISSIONS_FOLDER,label_file), 'rb') as f:
#                 full_results = pickle.load(f)
#             # return full_results['tokens'], full_results['token_indices'],full_results['emissions']    
#         emission_statistics.append(time.time() - start_batch)
#         if len(emission_statistics) % 31 == 30:
#             mean_time = np.array(emission_statistics).mean()
#             print('emission extraction average of {:.5f}m {:.5f}s'.format(mean_time // 60, mean_time % 60))





    

def main():


    save_emissions_to_file(MMS_EMISSIONS_FOLDER)



    #score_word = get_prob_word(trellis, norm_trellis, emission, tokens, token_indices, 3, times=(200, 220))
    #print("score word: ", score_word)


if __name__ == "__main__":
    main()