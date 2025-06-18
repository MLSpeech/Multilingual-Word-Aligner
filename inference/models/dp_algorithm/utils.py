import os
import torch
from .extract_features import embedding_distance_by_tolerence, Features_DP





def create_tensor_from_indices_masked(y, predictions_vector_size):
    tensor = torch.zeros(predictions_vector_size, dtype=torch.float32)
    for index in y:
        tensor[index] = 1
    return tensor

def find_optimal_positions_with_penalty(n, k, model_probabilities, sentence, embeddings, token_indices, emissions, **configuration): 
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
    w, max_tolerence, penalty_gap = torch.tensor(configuration['w']), configuration['max_tolerence'], configuration['penalty_gap']
    features_object = Features_DP(configuration['dp_features'])
    num_words_in_sentence = sentence.count("|") + 1
    # Step 1: Initialize DP table
    dp = torch.full((n + 1, k + 1), -float('inf'))  # DP table of shape (n+1, k+1)
    dp[:, 0] = 0  # Set the first column (column 0) to zero
    chosen_indices = torch.zeros((n + 1, k + 1), dtype=torch.int)  # DP table of shape (n+1, k+1)
    distances_arr = torch.zeros(n)
    
    # Step 2: Fill DP table with penalty for consecutive choices
    trace = torch.zeros((n + 1, k + 1), dtype=torch.int)  # To track decisions

    for frame in range (0, n):
        distances_arr[frame] = embedding_distance_by_tolerence(embeddings, y_indexes=[frame],
                            embedding_parts_min_max=(0,configuration['UnSupSeg_size']) ,max_tolerence=max_tolerence)[0]
    
    for j in range(1, k + 1):
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