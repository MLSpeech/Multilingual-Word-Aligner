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
import math
import pandas as pd
import numpy as np
from train.forcedAlignment.utils.constants import DEVICE
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional, Callable
from operator import itemgetter



def prepare_sentence(labels_dir, word_file):
    label_file = os.path.join(labels_dir, word_file)
    words = []
    # Read the file line by line
    with open(label_file, 'r') as file:
        for line in file:
            # Split each line into parts (assuming whitespace separates them)
            parts = line.split()
            if len(parts) >= 3:
                # The end time is the second element converted to an integer
                words.append(parts[2])
    sentence = "|".join(words).upper()
    return sentence


class STATISTICS_KEYS:
    TOKEN_KEY = 'token'
    MEAN_KEY = 'mean_token'
    STD_KEY = 'std_token'
    

def embedding_distance_by_tolerence(embedding, y_indexes, embedding_parts_min_max ,max_tolerence=4):
    #suppose embedding is [frames, embedding size]
    # y_indexes is the indexes we want to calculate distance - [3,20,30] etc..
    #taking each index pad it from 2 sides with the tolerence, clip to max frames and 0 and drop uniques
    y_index_padded = torch.tensor(np.array([
        np.unique(np.concatenate(
        [(y_index + np.array(list(range(1, max_tolerence+1)))) ,
        (y_index - np.array(list(range(1, max_tolerence+1)))) 
        ]).clip(0,len(embedding)-1))
        for y_index in y_indexes
    ]))
    embedding_indexes = torch.tensor(range(embedding_parts_min_max[0], embedding_parts_min_max[1]))
    compare_vectors = embedding[y_index_padded][:, :, embedding_indexes]  # Shape (num of indexes,distances for index,,embedding size)

    # Check the shape of compare_vectors to handle the possible issue with reverse slicing
    if compare_vectors.shape[1] > 1:
        compare_vectors_rotated = compare_vectors[:, torch.arange(compare_vectors.shape[1] - 1, -1, -1), :]#compare_vectors[:, ::-1, :]
    else:
        compare_vectors_rotated = compare_vectors  # No rotation if there's only one frame
    distances = torch.linalg.norm(compare_vectors - compare_vectors_rotated, axis=2)[:,:max_tolerence].sum(axis=1)

    return distances


# TODO: talk about the if condition if the first frame
def score_feature_word(predictions, y):
    if y[0] == 0:
        score = 1 + predictions[y[1]] # for the first index the probability is around 0 and we want that DP will think that this is a start of a word
    else:
        score = predictions[y[0]] + predictions[y[1]]
    
    # Check if the denominator is zero
    denominator = y[1] - y[0] - 1
    if denominator == 0:
        mean_scores_of_middle = 0
    else:
        mean_scores_of_middle = 1 - ((torch.sum(predictions[y[0] + 1 : y[1]])) / denominator)
    
    score += mean_scores_of_middle
    return score

def score_feature_frame(predictions,y):
    return predictions[y[1]]



def load_token_statistics(statistics_file):
    tokens_info = pd.read_csv(statistics_file)
    tokens_dict = tokens_info.set_index(STATISTICS_KEYS.TOKEN_KEY).to_dict('index')
    return tokens_dict

def word_mean_std(sentence, word_number, tokens_dict, extract_token_func=list):
    words = sentence.split('|')
    word = words[word_number - 1]
    tokens = extract_token_func(word)
    
    mean, std = 0, 0
    for token in tokens:
        if token == "'":
            continue
        mean += tokens_dict[token][STATISTICS_KEYS.MEAN_KEY]
        std += (tokens_dict[token][STATISTICS_KEYS.STD_KEY] ** 2)
    
    std_sqrt = math.sqrt(std)
    return mean, std_sqrt


def mms_word_emission_score(token_indices, word_number, emissions, y):
    word_indexes = token_indices[word_number - 1]
    start_frame, end_frame = y[0], y[1]
    denominator = 1 + end_frame - start_frame 
    if denominator == 0:
        emissions_score = 0
    else:
        emissions_score = emissions[start_frame:end_frame+1, word_indexes].sum() /denominator / len(word_indexes)

    return emissions_score

def normal_feature(y, mean, std):
    """Optimized version of normal_feature that avoids reading the CSV every time."""
    i = 0
    frame_duration = y[i+1] - y[i]

    total_normal_score = norm.pdf(frame_duration, loc=mean, scale=std)
    total_rate = (frame_duration / mean)
    return total_normal_score #, total_rate

                
def score_Dseg(embeddings, start_end_indices,embedding_parts_min_max):
    embedding_indexes = torch.tensor(np.array(range(embedding_parts_min_max[0], embedding_parts_min_max[1])))
    tzvia_vectors = embeddings[start_end_indices[1], embedding_indexes]
    score = float(tzvia_vectors.sum())
    return score


@dataclass
class Feature():
    score_function: Optional[Callable[[str],list]] = None
    settings: dict = field(default_factory=dict)
    

class Features_DP():
    # score feature - feature - function getting parameters returning score or two
    # we get list of keys map to function returning score
    # in the end we want list of scores,

    features_name_mapping = {
        'normal_feature': Feature(score_function=normal_feature,
                        settings={'letter_feature': ['start_end_indices','mean','std']}),
        'score_frame' : Feature(score_function=score_feature_frame,
                        settings={'mms_word': ['mms_probabilities', 'start_end_indices'], 'feature_word': ['probabilities', 'start_end_indices']}),
        'score_word': Feature(score_function=score_feature_word,
                        settings={'mms_word': ['mms_probabilities', 'start_end_indices'], 'feature_word': ['probabilities', 'start_end_indices']}),
        'distance_score': Feature(score_function=lambda distance_score: distance_score,
                        settings={'distance': ['distance_score']}),
        'mms_emission_score': Feature(score_function=mms_word_emission_score,
                        settings={'emission_score': ['words_tokens','word_number','emissions','start_end_indices']}),
        'dseg_score': Feature(score_function=score_Dseg,
                        settings={'dseg_score': ['embeddings','start_end_indices', 'embedding_indices']})
        #the args here need to be in order of them in the function(arg a, arg b, etc..)
    }
    
    
    def __init__(self, functions_to_run):
        self.functions_to_run = functions_to_run
        for function_name in functions_to_run: 
            func_keys = function_name.split(".")
            if func_keys[0] not in Features_DP.features_name_mapping:
                raise Exception(f"send function not in mapped functions got {func_keys[0]}")
            elif len(func_keys)>1 and func_keys[1] not in Features_DP.features_name_mapping[func_keys[0]].settings:
                raise Exception(f"send function not in mapped functions for {func_keys}")
            else:
                continue
        self.scores_statistics = {}
        for function_name in functions_to_run: 
            self.scores_statistics[function_name] = []

    def run_functions(self, args_dict):
        
        score_obj = {'scores':[], 'names':[]}
        for function_name in self.functions_to_run:
            func_keys = function_name.split(".")
            feature_object = Features_DP.features_name_mapping[func_keys[0]]
            if len(func_keys)>1:
                feature_keys = feature_object.settings[func_keys[1]]
            else:
                feature_keys = feature_object.settings[list(feature_object.settings.keys())[0]]
            try:
                values = itemgetter(*feature_keys)(args_dict)
                if type(values)!= tuple:
                    values = (values,)
            except Exception as e:
                raise Exception(f'required keys dont fit the dictionary got dict of: {args_dict.keys()} and keys:{feature_keys} for {function_name} ')
            score_for_function = feature_object.score_function(*values)
            score_obj['scores'].append(score_for_function)
            score_obj['names'].append(function_name)
                
        score_obj['scores'] = torch.tensor(score_obj['scores'])
        return score_obj

def get_predictions_masked(model, embeddings, labels, masks, conf_model, device=DEVICE):
    model.eval()
    with torch.no_grad():
        embedding_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE) #shape: torch.Size([5, 80, 192])
        masks_tensor = torch.tensor(np.array(masks)).to(DEVICE) #shape: torch.Size([5, 80])
        labels_tensor = torch.tensor(np.array(labels)).to(DEVICE) #shape: torch.Size([5, 80])
        outputs = model(input_ids=embedding_tensor) #shape: torch.Size([5, 80, 1])
        if conf_model["model_type"] == 'vgg':
            middle_index = embedding_tensor.shape[1] // 2
            embedding_tensor = embedding_tensor[:, middle_index:middle_index+1, :]  # Shape becomes [X, 1, Y]
        # Apply the mask to filter the tensors by indexing
        masked_embeddings = embedding_tensor[masks_tensor.bool()] #shape:([356, 192])
        masked_labels = labels_tensor[masks_tensor.bool()] #shape:([356])
        masked_outputs = outputs[masks_tensor.bool()] #shape:([356, 1])
        # Add the batch dimension back
        masked_embeddings = masked_embeddings.unsqueeze(0) #shape:([1, 356, 192])
        masked_labels = masked_labels.unsqueeze(0) #shape:([1, 356])
        masked_outputs = masked_outputs.unsqueeze(0)
        probabilities = torch.sigmoid(masked_outputs).view(-1).cpu() #shape:(356,)
        
    return masked_embeddings, probabilities, masked_labels



    
def main():
    pass
if __name__ == '__main__':
    main()