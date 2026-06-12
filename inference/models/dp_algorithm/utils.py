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
    max_tolerence, penalty_gap = configuration['max_tolerence'], configuration['penalty_gap']
    # Reuse pre-built objects when available (set once in align_wav.py before the file loop)
    features_object = configuration.get('_dp_features_obj') or Features_DP(configuration['dp_features'])
    w_floats = configuration.get('_w_floats') or list(configuration['w'])
    num_words_in_sentence = sentence.count("|") + 1

    # Step 1: Initialize DP table
    dp = torch.full((n + 1, k + 1), -float('inf'))
    dp[:, 0] = 0
    chosen_indices = torch.zeros((n + 1, k + 1), dtype=torch.int)

    # Step 2: Pre-compute per-frame distances in one batched call (vectorized)
    trace = torch.zeros((n + 1, k + 1), dtype=torch.int)
    distances_arr = embedding_distance_by_tolerence(
        embeddings, y_indexes=list(range(n)),
        embedding_parts_min_max=(0, configuration['UnSupSeg_size']),
        max_tolerence=max_tolerence,
    )

    # Shared dict; only start_end_indices, distance_score, word_number change per iteration
    args_for_features = {
        'probabilities': model_probabilities,
        'words_tokens': token_indices,
        'emissions': emissions,
        'embeddings': embeddings,
        'start_end_indices': None,
        'distance_score': None,
        'word_number': None,
    }

    for j in range(1, k + 1):
        min_frame = penalty_gap * j
        max_frame = n + 1 - penalty_gap * (num_words_in_sentence - j)
        args_for_features['word_number'] = j
        for i in range(1, n + 1):
            # Option 1: Don't place a 1 at position i
            dp[i][j] = dp[i - 1][j]
            chosen_indices[i][j] = chosen_indices[i - 1][j]

            # Option 2: Place a 1 at position i, with a penalty gap
            if i > min_frame and i < max_frame:
                last_index_we_chose = chosen_indices[i - penalty_gap - 1][j - 1]
                args_for_features['start_end_indices'] = [last_index_we_chose, i - 1]
                args_for_features['distance_score'] = distances_arr[i - 1]

                total_score_for_index = (
                    features_object.run_functions_weighted(args_for_features, w_floats)
                    + dp[i - penalty_gap - 1][j - 1]
                )

                if total_score_for_index > dp[i][j]:
                    dp[i][j] = total_score_for_index
                    trace[i][j] = 1
                    chosen_indices[i][j] = i - 1

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