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
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import json
import torch
from train.forcedAlignment.utils.constants import DEVICE, DATASETS_MAPPING, DATASET, LABELS_DIR_VAL, DP_PATHES, MODEL_TRAINED_DATASET, MODEL_NAME, DEVICE, models_configurations
from train.models.utils import load_model
from train.forcedAlignment.utils.preprocess import prepare_dataset
from train.forcedAlignment.dynamic_prog.extract_features import Features_DP, prepare_sentence, get_predictions_masked, embedding_distance_by_tolerence
from train.models.MMS.mms_DP import extract_file_emissions_token, get_emmissions_not_norm
from train.heat_map.heat_maps_ploter import dp_results_textgrid, extract_textgrid_results_to_image_dp, plot_heatmaps, plot_emissions
from train.models.UnsupSeg.predict import unsupseg_pred
from train.models.MMS.MMS_details import prepare_mms_pred



def save_heatmap(array_2d, save_path, dpi=600, cmap='viridis'):
    """
    Transpose a 2D numpy array and plot it as a high-resolution heatmap with custom axis labels.
    
    Parameters:
    - array_2d: 2D numpy array to plot
    - save_path: Path to save the image (include file extension like .png or .jpg)
    - dpi: Dots per inch for resolution (default: 600, can be increased for higher quality)
    """
    # Transpose the array
    transposed_array = np.transpose(array_2d)
    
    # Create figure with high resolution and tight layout
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    fig.tight_layout(pad=2.0)
    
    # Create heatmap with inferno colormap
    heatmap = ax.imshow(transposed_array, 
                       aspect='auto', 
                       cmap=cmap,
                       interpolation='nearest')  # Avoid blurring
    
    # Colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    
    # X-axis (every 50 frames)
    x_ticks = np.arange(0, transposed_array.shape[1], 50)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=8)
    ax.set_xlabel('Frames (10 msec intervals)', fontsize=10)
    
    # Y-axis (first/middle/last)
    y_ticks = [0, transposed_array.shape[0]//2, transposed_array.shape[0]-1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=8)
    ax.set_ylabel('')
    
    # Save as PNG (removed quality parameter)
    if not save_path.endswith('.png'):
        save_path += '.png'  # Force PNG extension
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"Saved inferno heatmap to {save_path} (DPI={dpi})")


def save_heatmap_embeddings(array_2d, save_path, dpi=600, cmap='viridis'):
    """
    Transpose a 2D numpy array and plot as a high-res heatmap with custom y-axis text.
    """

    # Transpose the array
    transposed_array = np.transpose(array_2d)
    
    # Create figure with tighter layout
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    fig.subplots_adjust(left=0.15)  # Reduced left margin
    
    # Heatmap
    heatmap = ax.imshow(transposed_array, 
                       aspect='auto', 
                       cmap=cmap,
                       interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    
    # X-axis
    x_ticks = np.arange(0, transposed_array.shape[1], 50)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=8)
    ax.set_xlabel('Frames (10 msec intervals)', fontsize=10)
    
    # Y-axis ticks (hidden but used for positioning)
    y_ticks = [0, transposed_array.shape[0]//2, transposed_array.shape[0]-1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=8)
    ax.set_ylabel('')
    
    # Calculate perfect vertical positions
    y_mid = transposed_array.shape[0]//2
    first_label_pos = y_mid/2  # Centered in first half
    second_label_pos = y_mid*1.5  # Centered in second half
    
    # First label (perfectly centered two-line text)
    ax.text(-0.1, first_label_pos, 'UnSupSeg\nCNN', 
            transform=ax.get_yaxis_transform(),
            ha='center', va='center',  # Centered alignment
            fontsize=10, color='black',
            linespacing=1.1,
            bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
    
    # Second label
    ax.text(-0.1, second_label_pos, 'MMS', 
            transform=ax.get_yaxis_transform(),
            ha='center', va='center',
            fontsize=10, color='black',
            bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
    
    # Save
    if not save_path.endswith('.png'):
        save_path += '.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"Saved final heatmap to {save_path}")



def load_config_and_weights(config_path):
    # Load the config data from the JSON file
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    
    # Extract the final_weights and convert from list back to Torch tensor
    final_weights = torch.from_numpy(np.array(train_config['final_weights']))

    return train_config, final_weights

def load_DP(dp_path):
    config_path = os.path.join(dp_path, f'config.json')
    dp_cfg, w = load_config_and_weights(config_path)
    features_object = Features_DP(dp_cfg['features'])
    
    return dp_cfg, w, features_object

def find_optimal_positions_with_penalty(n, k, model_probabilities, sentence, embeddings, w, max_tolerence=4, penalty_gap=5, features_object=None, token_indices=None, emissions=None): 
    """
    Find the optimal positions of 1s in a binary sequence with a penalty for consecutive selections.

    Args:
        n (int): Length of the sequence.
        k (int): Number of 1s to place.
        probs (torch.Tensor): Probabilities of placing 1 at each position (shape: [n]).
        penalty_gap (int): Minimum gap between consecutive 1s.

    Returns:
        positions (list): Indices of the optimal positions for the 1s.
        max_prob (float): Maximum achievable probability.
    """
    num_words_in_sentence = sentence.count("|") + 1
    k = num_words_in_sentence
    # Step 1: Initialize DP table
    dp = torch.full((n + 1, k + 1), -float('inf'))  # DP table of shape (n+1, k+1)
    dp[:, 0] = 0  # Set the first column (column 0) to zero
    chosen_indices = torch.zeros((n + 1, k + 1), dtype=torch.int)  # DP table of shape (n+1, k+1)
    distances_arr = torch.zeros(n)
    dpss = {'score_frame': torch.zeros((n + 1, k + 1)), 
        'mms_emission_score': torch.zeros((n + 1, k + 1)),
        'score_word': torch.zeros((n + 1, k + 1)),
        'distance_score': torch.zeros((n + 1, k + 1))}
    
    dpss_w = {'score_frame': torch.zeros((n + 1, k + 1)), 
        'mms_emission_score': torch.zeros((n + 1, k + 1)),
        'score_word': torch.zeros((n + 1, k + 1)),
        'distance_score': torch.zeros((n + 1, k + 1))}
    a = True
    
    # Step 2: Fill DP table with penalty for consecutive choices
    trace = torch.zeros((n + 1, k + 1), dtype=torch.int)  # To track decisions

    for frame in range (0, n):
        distances_arr[frame] = embedding_distance_by_tolerence(embeddings, y_indexes=[frame],
                            embedding_parts_min_max=(0,DATASETS_MAPPING[DATASET]['UnSupSeg_size']) ,max_tolerence=max_tolerence)[0]
    
    for j in range(1, k + 1):
        #word_mean, word_std = word_mean_std(sentence, word_number=j, tokens_dict=tokens_dict) # Normal feature
        min_frame = penalty_gap*j
        max_frame = n + 1 - penalty_gap*(num_words_in_sentence-j)
        for i in range(1, n + 1):
            
            # Option 1: Don't place a 1 at position i
            dp[i][j] = dp[i - 1][j]
            chosen_indices[i][j] = chosen_indices[i - 1][j]
            
            # Option 2: Place a 1 at position i, with a penalty gap
            if i > min_frame and i < max_frame:
                last_index_we_chose = chosen_indices[i - penalty_gap - 1][j - 1]
                #calculate features
                distances = distances_arr[i-1]
                
                args_for_features = {'start_end_indices':[last_index_we_chose, i-1],
                                     "probabilities":model_probabilities, 'distance_score': distances,
                                     'words_tokens':token_indices,'word_number':j,'emissions':emissions,
                                     'embeddings':embeddings
                                     }
                
                score_object = features_object.run_functions(args_for_features)
                dpss['score_frame'][i][j] = score_object['scores'][0]
                dpss['mms_emission_score'][i][j] = score_object['scores'][1]
                dpss['score_word'][i][j] = score_object['scores'][2]
                dpss['distance_score'][i][j] = score_object['scores'][3]

                dpss_w['score_frame'][i][j] = score_object['scores'][0]*w[0]
                dpss_w['mms_emission_score'][i][j] = score_object['scores'][1] *w[1]
                dpss_w['score_word'][i][j] = score_object['scores'][2]*w[2]
                dpss_w['distance_score'][i][j] = score_object['scores'][3]*w[3]
                if a:
                    a = False
                    print(score_object)
                    print(dpss['score_frame'][i][j], dpss['mms_emission_score'][i][j], dpss['score_word'][i][j], dpss['distance_score'][i][j])
                total_score_for_index = (w*score_object['scores']).sum() + dp[i - penalty_gap - 1][j - 1]

                if total_score_for_index > dp[i][j]:
                    dp[i][j] = total_score_for_index
                    
                    trace[i][j] = 1
                    chosen_indices[i][j] = i-1

    # Step 3: Backtrack to find positions
    positions = []
    i, remaining_1s = n, k

    while remaining_1s > 0:
        if int(trace[i][remaining_1s]) == 1:
            positions.append(i - 1)
            remaining_1s -= 1
            i -= penalty_gap  # Skip penalty_gap positions
        else:
            i -= 1
        
    
    positions.reverse()
    max_prob = dp[n][k].item()

    return positions, dp, dpss


def main():
    PLOT_EMBEDDINGS = False
    PLOT_FEATURES_HEATMAPS = False
    PLOT_GRAPHS = False
    PLOT_EMISSIONS = True


    file_name = 'dr6_mcmj0_sx14.wrd'
    #for dynamic programming we need - 
    # for letter prediction - sentence + statistics files - each time we choose one we check in which word we are and calculate score
    # for score of model - we need model probabilities
    # for distances we need embeddings and specific place
    file_name_without_extension = file_name.rsplit('.', 1)[0]
    path = os.path.join(project_dir, 'heat_map', 'heatmaps_plots', file_name_without_extension)

    
    dp_cfg, w, features_object = load_DP(DP_PATHES[MODEL_TRAINED_DATASET][MODEL_NAME])
    model_path = dp_cfg['model_path']
    device = DEVICE
    model, conf_model = load_model(model_path, device)
    embeddings_val, labels_val, masks_val = prepare_dataset(labels_dir=LABELS_DIR_VAL, mode='val', model_arguments=conf_model, specific_file=file_name)
    if not embeddings_val:
        print("error")
    masked_embeddings_val, probabilities_val, masked_labels_val = get_predictions_masked(model, embeddings_val, labels_val, masks_val, conf_model)
    sentence = prepare_sentence(LABELS_DIR_VAL, file_name)
    print(sentence)
    _, token_indices, emissions = extract_file_emissions_token(LABELS_DIR_VAL, file_name)
    num_words_in_sentence = sentence.count("|") + 1
    y_pred, dp, dpss = find_optimal_positions_with_penalty(len(masked_embeddings_val[0]), num_words_in_sentence, probabilities_val, sentence, masked_embeddings_val[0].cpu(), w, dp_cfg['max_tolerence'], dp_cfg['penalty_gap'], features_object, token_indices, emissions)
    yi = (masked_labels_val == 1).nonzero(as_tuple=True)[1].cpu().numpy()

    if torch.sum(masked_labels_val == 1) < num_words_in_sentence:
        y_pred = y_pred[:-1]

    masked_labels_val = masked_labels_val[0].cpu().numpy()
    dynamic_preds_10_vector = np.zeros(masked_labels_val.shape[0])
    dynamic_preds_10_vector[y_pred] = 1
    probabilities_val = probabilities_val.numpy()
    predictions = (probabilities_val >= 0.5).astype(float)
    
    # MMS bounadries predictions
    mms_folder = models_configurations['mms']['val_folder']
    mms_pred = prepare_mms_pred(file_name_without_extension, mms_folder)
    mms_pred_10ms_win = [x * 2 for x in mms_pred]
    mms_pred_10ms_win = [int(x) for x in mms_pred_10ms_win]
    mms_preds_10ms_vector = np.zeros(masked_labels_val.shape[0])
    mms_preds_10ms_vector[mms_pred_10ms_win] = 1

    # UnSupSeg bounadries predictions and probabilities
    wav_file_name = file_name.replace('.wrd', '.wav')
    wav_path = os.path.join(LABELS_DIR_VAL, wav_file_name)
    unsupseg_probabilities, unsupseg_preds = unsupseg_pred(wav_path)
    unsupseg_probabilities = unsupseg_probabilities.cpu().detach().numpy().squeeze()
    unsupseg_preds_vector = np.zeros(masked_labels_val.shape[0])
    unsupseg_preds_vector[unsupseg_preds] = 1

        
    #visualize_of the results
    if PLOT_FEATURES_HEATMAPS:
        plot_heatmaps(path, dpss)
    
    if PLOT_GRAPHS:
        word_posix = DATASETS_MAPPING[DATASET]['word_posix']
        textgrid_file_path = dp_results_textgrid(file_name.replace(word_posix,''), masked_labels_val, probabilities_val, predictions, dynamic_preds_10_vector, mms_preds_10ms_vector, unsupseg_preds_vector, unsupseg_probabilities, path, units=10) 
        wav_file = os.path.join(LABELS_DIR_VAL, file_name.replace(word_posix,'.wav'))
        extract_textgrid_results_to_image_dp(textgrid_path=textgrid_file_path, wav_file=wav_file, embedding=masked_embeddings_val[0].cpu().numpy(), dp=dp)

    if PLOT_EMBEDDINGS:
        save_path = os.path.join(path, 'embeddings.png')
        save_heatmap_embeddings(masked_embeddings_val[0].cpu().numpy(), save_path)

    if PLOT_EMISSIONS:
        emissions_not_norm = get_emmissions_not_norm(LABELS_DIR_VAL, file_name)
        save_path = os.path.join(path, 'emissions.png')
        plot_emissions(emissions_not_norm.cpu().numpy(), save_path)
        


if __name__ == '__main__':
    main()