
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


    

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
    #ref_vectors = embedding[y_indexes][:, None, embedding_indexes]  # Shape (num of indexes,vector,embedding size)
    compare_vectors = embedding[y_index_padded][:, :, embedding_indexes]  # Shape (num of indexes,distances for index,,embedding size)
    # Check the shape of compare_vectors to handle the possible issue with reverse slicing
    if compare_vectors.shape[1] > 1:
        compare_vectors_rotated = compare_vectors[:, torch.arange(compare_vectors.shape[1] - 1, -1, -1), :]#compare_vectors[:, ::-1, :]
    else:
        compare_vectors_rotated = compare_vectors  # No rotation if there's only one frame
    distances = torch.linalg.norm(compare_vectors - compare_vectors_rotated, axis=2)[:,:max_tolerence].sum(axis=1)
    return distances


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


def mms_word_emission_score(token_indices, word_number, emissions, y):
    word_indexes = token_indices[word_number - 1]
    start_frame, end_frame = y[0], y[1]
    denominator = 1 + end_frame - start_frame 
    if denominator == 0:
        emissions_score = 0
    else:
        emissions_score = emissions[start_frame:end_frame+1, word_indexes].sum() /denominator / len(word_indexes)
    return emissions_score


# score feature - feature - function getting parameters returning score or two
# we get list of keys map to function returning score
# in the end we want list of scores,
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import inspect
from operator import itemgetter

@dataclass
class Feature():
    score_function: Optional[Callable[[str],list]] = None
    settings: dict = field(default_factory=dict)
    

class Features_DP():
    features_name_mapping = {
        'score_frame' : Feature(score_function=score_feature_frame,
                        settings={'mms_word': ['mms_probabilities', 'start_end_indices'], 'feature_word': ['probabilities', 'start_end_indices']}),
        'score_word': Feature(score_function=score_feature_word,
                        settings={'mms_word': ['mms_probabilities', 'start_end_indices'], 'feature_word': ['probabilities', 'start_end_indices']}),
        'distance_score': Feature(score_function=lambda distance_score: distance_score,
                        settings={'distance': ['distance_score']}),
        'mms_emission_score': Feature(score_function=mms_word_emission_score,
                        settings={'emission_score': ['words_tokens','word_number','emissions','start_end_indices']}),
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

    def _resolve_feature(self, function_name):
        func_keys = function_name.split(".")
        feature_object = Features_DP.features_name_mapping[func_keys[0]]
        if len(func_keys) > 1:
            feature_keys = feature_object.settings[func_keys[1]]
        else:
            feature_keys = feature_object.settings[list(feature_object.settings.keys())[0]]
        return feature_object.score_function, feature_keys

    def run_functions(self, args_dict):
        score_obj = {'scores': [], 'names': []}
        for function_name in self.functions_to_run:
            fn, feature_keys = self._resolve_feature(function_name)
            try:
                values = itemgetter(*feature_keys)(args_dict)
                if not isinstance(values, tuple):
                    values = (values,)
            except Exception as e:
                raise Exception(f'required keys dont fit the dictionary got dict of: {args_dict.keys()} and keys:{feature_keys} for {function_name} ')
            score_obj['scores'].append(fn(*values))
            score_obj['names'].append(function_name)
        score_obj['scores'] = torch.tensor(score_obj['scores'])
        return score_obj

    def run_functions_weighted(self, args_dict, w_floats):
        """Compute weighted sum of feature scores directly, avoiding intermediate tensor allocation."""
        total = 0.0
        for i, function_name in enumerate(self.functions_to_run):
            fn, feature_keys = self._resolve_feature(function_name)
            try:
                values = itemgetter(*feature_keys)(args_dict)
                if not isinstance(values, tuple):
                    values = (values,)
            except Exception as e:
                raise Exception(f'required keys dont fit the dictionary got dict of: {args_dict.keys()} and keys:{feature_keys} for {function_name} ')
            total = total + w_floats[i] * fn(*values)
        return total


