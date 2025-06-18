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
import random
import math
import time
import numpy as np
import wandb
from train.forcedAlignment.utils.constants import LABELS_DIR, MODEL_NAME, LABELS_DIR_VAL, DATASETS_MAPPING, DATASET, OUTPUT_DP_LOG_DIR, TIME, MODEL_PATHS, RUN_EXPERIMENTS
from train.forcedAlignment.utils.preprocess import prepare_dataset, get_all_model_files_names, get_labels_in_sec
from train.models.utils import load_model
from train.forcedAlignment.dynamic_prog.extract_features import Features_DP, get_predictions_masked, embedding_distance_by_tolerence, prepare_sentence
from train.evaluation.compare_results import score_boundaries_DP
from train.forcedAlignment.utils.logger_utils import setup_logging_DP
from train.models.MMS.mms_DP import extract_file_emissions_token
from train.forcedAlignment.dynamic_prog.dynamic_prog_statistics import calculate_statistics, log_best_statistics
from train.forcedAlignment.dynamic_prog.loss_DP import LossFunction_DP
from train.forcedAlignment.dynamic_prog.DP_utils import wandb_log_statistics, save_config_and_weights, create_batches



def phi(embeddings, probabilities, sentence, y, max_tolerence=4, features_object=None, token_indices=None, emissions=None):
    last_index_we_chose = 0
    word_number = 1
    scores = torch.zeros(len(features_object.functions_to_run))
    for pred in y:
        #word_mean, word_std = word_mean_std(sentence, word_number=word_number, tokens_dict=tokens_dict)
        if pred >= len(embeddings):
            continue
        distances = embedding_distance_by_tolerence(embeddings, y_indexes=[pred],
                    embedding_parts_min_max=(0,DATASETS_MAPPING[DATASET]['UnSupSeg_size']) ,max_tolerence=max_tolerence)[0]
        
        args_for_features = {'start_end_indices':[last_index_we_chose, pred], "probabilities":probabilities,
                                     'distance_score': distances, 'words_tokens':token_indices,'word_number':word_number,'emissions':emissions
                                     ,'embeddings':embeddings
                                     }
        score_object = features_object.run_functions(args_for_features)
        scores+=score_object['scores']
        last_index_we_chose = pred
        word_number += 1
    return scores


def update_measures_by_validation_evaluation(labels_files_val, conf_model, model, train_cfg, w, features_object, mode='val', tol_lst=[0,2,4,9], labels_dir=LABELS_DIR_VAL):
    measures = {tol: {'TP': 0, 'TP+FP': 0, 'TP+FN': 0} for tol in tol_lst}
    
    for val_file in labels_files_val:
        # Preparing the data
        embeddings_val, labels_val, masks_val = prepare_dataset(labels_dir=labels_dir, mode=mode, model_arguments=conf_model, specific_file=val_file)
        labels_in_sec = get_labels_in_sec(labels_dir, val_file)
        if not embeddings_val:
            continue
        masked_embeddings_val, probabilities_val, masked_labels_val = get_predictions_masked(model, embeddings_val, labels_val, masks_val, conf_model)
        sentence_val = prepare_sentence(labels_dir, val_file)
        _, token_indices, emissions = extract_file_emissions_token(labels_dir, val_file)
        num_words_in_sentence = sentence_val.count("|") + 1

        y_pred, _, _ = find_optimal_positions_with_penalty(len(masked_embeddings_val[0]), num_words_in_sentence, probabilities_val, sentence_val, masked_embeddings_val[0].cpu(), w, train_cfg['max_tolerence'], train_cfg['penalty_gap'], features_object, token_indices, emissions) #masked_labels_val[0].sum()


        # Calculate measurments in tolerence window for validation file
        for tol in tol_lst:
            tp, tp_plus_fp, tp_plus_fn = score_boundaries_DP(labels_in_sec, y_pred, tol)
            measures[tol]['TP'] += tp
            measures[tol]['TP+FP'] += tp_plus_fp
            measures[tol]['TP+FN'] += tp_plus_fn
    
    return measures

        
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

    return positions, max_prob, dp




# Training function based on the iterative algorithm
def train_phoneme_alignment(train_cfg, labels_files, labels_files_val):
    model, conf_model = load_model(train_cfg['model_path'])
    train_cfg.update(conf_model)
    logger, _, log_dir = setup_logging_DP(train_cfg, TIME, model_args=conf_model, run_exp=RUN_EXPERIMENTS, optuna=False)
    # Initialize weight vector w to zero
    features_object = Features_DP(train_cfg['features'])
    w = torch.zeros(len(features_object.functions_to_run))
    loss_function = LossFunction_DP(w, train_cfg['C'], features_object, train_cfg)
    # Initialize WandB run
    if RUN_EXPERIMENTS: # Meaning runung some testing and dont want to upload the results!
        wandb_log_mode = "online"
    else:
        wandb_log_mode = "offline"
    if train_cfg['early_stop']:
        run_name = f"{train_cfg['model_type']}_early_stop_{TIME}"
    else:
        run_name = f"{train_cfg['model_type']}_no_early_stop_{TIME}"
    wandb.init(project="Word Alignment Roy&Meidan", 
               config=train_cfg, 
               mode=wandb_log_mode,
               name=run_name)

    since = time.time()
    files_per_batch=150
    num_batches = math.ceil(len(labels_files)/files_per_batch)
    batch_num=0
    best_w  = w.clone()
    best_statistics = {}
    best_accuracy_tol_0 = 0
    num_iteration_with_no_change = 0
    batch_train_files = 0
    num_train_file = 0
    # Split files into batches
    train_batches = create_batches(labels_files, files_per_batch)
    #for train_batch, val_batch in zip(train_batches, val_batches):
    for epoch in range(train_cfg['num_epochs_dp']):
        logger.warning(f"{'-'*(50)} Epoch {epoch+1} / {train_cfg['num_epochs_dp']}  {'-'*(50)}")
        start_epoch= time.time()
        for train_batch in train_batches:
            batch_num += 1
            logger.warning(f"{'-'*(50)} Batch {batch_num} / {num_batches}  {'-'*(50)}")
           
            start_batch= time.time()
            if num_iteration_with_no_change >= 300:
                w = best_w.clone()
                logger.warning(f"Stuck w, return to best w!")
            
            for train_file in train_batch:
                num_train_file += 1
                # Preparing the data
                embeddings, labels, masks = prepare_dataset(labels_dir=LABELS_DIR, mode='train', model_arguments=conf_model, specific_file=train_file)
                if not embeddings:
                    continue
                masked_embeddings, probabilities, masked_labels = get_predictions_masked(model, embeddings, labels, masks, conf_model)
                sentence = prepare_sentence(LABELS_DIR, train_file)
                num_words_in_sentence = sentence.count("|") + 1
                
                _, token_indices, emissions = extract_file_emissions_token(LABELS_DIR, train_file)
    
                # Find Prediction base on DP
                y_pred, _, _ = find_optimal_positions_with_penalty(len(masked_embeddings[0]), num_words_in_sentence, probabilities, sentence, masked_embeddings[0].cpu(), #masked_labels[0].sum()
                                                                w, train_cfg['max_tolerence'], train_cfg['penalty_gap'], features_object, token_indices, emissions)
                labels_in_sec = get_labels_in_sec(LABELS_DIR, train_file)
                labels_frames = np.array([int(math.floor(time_label * 100)) for time_label in labels_in_sec])
                yi = torch.tensor(labels_frames)

                phi_pred = phi(masked_embeddings[0].cpu(), probabilities, sentence, y_pred, train_cfg['max_tolerence'], features_object, token_indices, emissions)
                phi_i = phi(masked_embeddings[0].cpu(), probabilities, sentence, yi, train_cfg['max_tolerence'], features_object, token_indices, emissions)

                # Loss
                a = phi_i - phi_pred
                loss = loss_function.compute_loss(yi, y_pred, a)
                loss_function.update_weights(loss, a)
                batch_train_files += 1

                if RUN_EXPERIMENTS:
                    wandb.log({"loss": float(loss), "train_num": num_train_file})

                
            measures = update_measures_by_validation_evaluation(labels_files_val, conf_model, model, train_cfg, w, features_object, mode='val', tol_lst=[0,2,4,9])
            # Calculate measurments for all validation
            all_statistics = calculate_statistics(measures, [0,2,4,9], logger)
            wandb_log_statistics(all_statistics, batch_num)
            
            # Update w for best accuracy base on the validation
            if all_statistics[0]['Accuracy'] > best_accuracy_tol_0:
                best_accuracy_tol_0 = all_statistics[0]['Accuracy']
                best_w = np.copy(w)
                best_statistics = all_statistics

            logger.warning('batch complete in {:.0f}m {:.0f}s'.format((time.time() - start_batch) // 60, (time.time() - start_batch) % 60))
            logger.warning(f'w is: {w}')

        logger.warning('Epoch complete in {:.0f}h {:.0f}m {:.0f}s'.format((time.time() - start_epoch) // 3600, ((time.time() - start_epoch) % 3600) // 60, (time.time() - start_epoch) % 60))
    
    logger.warning('Train complete in {:.0f}h {:.0f}m {:.0f}s'.format((time.time() - since) // 3600, ((time.time() - since) % 3600) // 60, (time.time() - since) % 60))
    logger.warning(f'Best W is: {best_w}')
    log_best_statistics(best_statistics, logger)

    # Log the final statistics and plots to WandB
    if RUN_EXPERIMENTS:
        wandb.log({
            "best_w": best_w.tolist(),  
            "best_statistics": best_statistics,
        })
    # Finish the WandB run
    wandb.finish()

    return best_w, best_statistics, log_dir, train_cfg


# Define your prediction function (this is a placeholder)
def predict_alignment_DP(model, embeddings, labels, masks, w, labels_dir, file, conf_model, train_cfg, features_object):
    masked_embeddings, probabilities, masked_labels = get_predictions_masked(model, embeddings, labels, masks, conf_model)
    sentence = prepare_sentence(labels_dir, file)
    _, token_indices, emissions = extract_file_emissions_token(labels_dir, file)
    # Find Prediction base on DP
    num_words_in_sentence = sentence.count("|") + 1
    y_pred, _, _ = find_optimal_positions_with_penalty(len(masked_embeddings[0]), num_words_in_sentence, probabilities, sentence, masked_embeddings[0].cpu(), #masked_labels[0].sum()
                                                                w, train_cfg['max_tolerence'], train_cfg['penalty_gap'], features_object, token_indices, emissions)
    return y_pred



def main():

    model_name = MODEL_NAME
    early_stop = True
    num_epochs_dp = 1
    penalty_gap = 4
    max_tolerence = 1
    C=3
    comment = 'Train Conformer model no early stop'

    if early_stop:
        model_path = MODEL_PATHS[DATASET][model_name]['early_stop']
    else:
        model_path = MODEL_PATHS[DATASET][model_name]['no_early_stop']
    
    labels_files = get_all_model_files_names(labels_dir=LABELS_DIR)
    random.shuffle(labels_files)
    labels_files_val = get_all_model_files_names(labels_dir=LABELS_DIR_VAL)

    train_cfg = {
        'early_stop': early_stop,
        'num_epochs_dp': num_epochs_dp,
        'log_dir_DP': OUTPUT_DP_LOG_DIR,
        'C': C,
        'max_tolerence': max_tolerence,
        'penalty_gap': penalty_gap,
        'model_path': model_path,
        'comment': comment,
        'features': ['score_frame.feature_word','mms_emission_score','score_word.feature_word','distance_score']    
        }
    w_train, statistics, log_dir, full_config = train_phoneme_alignment(train_cfg, labels_files, labels_files_val)

    if RUN_EXPERIMENTS:
        save_config_and_weights(log_dir, full_config, w_train)

        
if __name__ == '__main__':
    main()