import torch
import matplotlib.pyplot as plt
import os
import numpy as np


def score_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the segmentation boundaries.

    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.

    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_boundaries_ref = 0
    n_boundaries_seg = 0
    n_boundaries_correct = 0
    
    for i_boundary, boundary_ref in enumerate(ref):

        boundary_seg = seg[i_boundary]

        boundary_ref = list(np.nonzero(boundary_ref)[0])
        boundary_seg = list(np.nonzero(boundary_seg)[0])
        n_boundaries_ref += len(boundary_ref)
        n_boundaries_seg += len(boundary_seg)

        for i_seg in boundary_seg:
            for i, i_ref in enumerate(boundary_ref):
                if abs(i_seg - i_ref) <= tolerance:
                    n_boundaries_correct += 1
                    boundary_ref.pop(i)
                    break

    return n_boundaries_correct, n_boundaries_seg, n_boundaries_ref

def score_boundaries_DP(labels, pred, tol):

    count_ones_labels = len(labels)
    count_ones_preds = len(pred)

    labels_time_sec = np.array(labels)
    pred_indices = np.array(pred)
    pred_time_sec = pred_indices*0.01

    # Calculate the distances (absolute difference between indices)
    distances = np.abs(labels_time_sec - pred_time_sec)

    # Count how many distances are <= tol
    time_tol = (tol + 1)*0.01
    count_within_tol = np.sum(distances <= time_tol)
    tp = count_within_tol
    tp_plus_fp = count_ones_labels
    tp_plus_fn = count_ones_labels

    return tp, tp_plus_fp, tp_plus_fn



def compare_predictions_with_labels(predictions, labels, window):
    if predictions.ndim == 3:
        # Flatten the array if it has shape (x, y, z)
        predictions_flatt = predictions.reshape(-1, 1)
    else:
        predictions_flatt = predictions

    # Check if labels is a list of tensors
    if isinstance(labels, list):
        if isinstance(labels[0], torch.Tensor):
            labels = labels[0]  # Extract the tensor from the list

    # No need to convert predictions to a tensor, since it's already a PyTorch tensor
    predictions_tensor = predictions_flatt

    # Pad the labels with zeros to match the length of predictions
    padding_length = len(predictions_flatt) - len(labels)
    if padding_length > 0:
        padded_labels = torch.cat((labels, torch.zeros(padding_length, device=labels.device)))
    else:
        padded_labels = labels

    # Create a copy of padded_labels that will be modified
    copy_padded_labels = padded_labels.clone()

    # Measurements
    count_1_labels = (padded_labels == 1).sum().item()
    count_window_hits = 0
    count_window_misses = 0
    count_window_over_hit = 0

    # Check predictions and count based on the conditions
    for j in range(len(predictions_tensor)):
        if predictions_tensor[j] > 0.5:
            # Define the window of 4 around j
            window_start = max(0, j - window)
            window_end = min(len(padded_labels), j + (window + 1))  # +1 for inclusive
            window_labels = padded_labels[window_start:window_end]
            copy_window_labels = copy_padded_labels[window_start:window_end]

            if 1 in window_labels and 1 in copy_window_labels:
                count_window_hits += 1
                # Set the corresponding window in copy_padded_labels to zero
                copy_padded_labels[window_start:window_end] = 0
            else:
                count_window_misses += 1

    return count_1_labels, count_window_hits, count_window_misses, count_window_over_hit


def calculate_real_model_measurement_1(probabilities, labels, max_window):

    percentage_hit_window = torch.zeros(max_window +1)
    percentage_miss_window = torch.zeros(max_window +1)
    percentage_over_hit_window = torch.zeros(max_window +1)


    for j in range(probabilities.shape[0]):
        for win in range(max_window+1):
            # Get the j-th slice of the NumPy array with shape (1, y, z)
            prob_slice = probabilities[j:j+1, :, :]  # Slicing to shape (1, y, z)
            
            # Get the j-th list and reshape to (1, y)
            labels_slice = [labels[j]]  # Wrap the inner list to shape (1, y)

            count_1_labels, count_window_hits, count_window_misses, count_window_over_hit = compare_predictions_with_labels(prob_slice, labels_slice, win)
            if count_1_labels == 0:
                count_1_labels = 1
            percentage_hit_window[win] += float(count_window_hits / count_1_labels)
            percentage_miss_window[win] += float(count_window_misses / count_1_labels)
            percentage_over_hit_window[win] += float(count_window_over_hit / count_1_labels)


    return percentage_hit_window, percentage_miss_window, percentage_over_hit_window

def evaluate_predictions_with_window(win, labels, predictions):
    """
    Evaluates predictions within a specified window for each label = 1.

    Args:
        win (int): The window size.
        labels (torch.Tensor): The ground truth labels (1D tensor).
        predictions (torch.Tensor): The predicted labels (1D tensor).

    Returns:
        float: The proportion of label=1 indices that have at least one prediction=1 in their window.
    """
    # Ensure tensors are 1D
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("Both labels and predictions must be 1D tensors.")
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    
    total_positive_labels = torch.sum(labels == 1).item()
    if total_positive_labels == 0:
        raise ValueError("No positive labels (label=1) found in the input.")

    correct_predictions = 0

    for idx in torch.nonzero(labels == 1, as_tuple=True)[0]:  # Indices where labels are 1
        start_idx = max(0, idx - win)
        end_idx = min(len(predictions), idx + win + 1)
        if torch.any(predictions[start_idx:end_idx] == 1):
            correct_predictions += 1

    return correct_predictions, total_positive_labels



def calculate_real_model_measurement(predictions, labels, mask):

    predictions = predictions.view(-1)
    labels = labels.view(-1)
    mask = mask.view(-1)
    predictions = predictions[mask.bool()]
    labels = labels[mask.bool()]
    
    # Calculate True Positives, False Positives, and False Negatives
    TP = torch.sum((predictions == 1) & (labels == 1)).item()
    FP = torch.sum((predictions == 1) & (labels == 0)).item()
    FN = torch.sum((predictions == 0) & (labels == 1)).item()
    
    # Precision, Recall, and F1-Score
    accuracy = torch.sum(predictions == labels).item() / len(labels.view(-1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    os = (recall / precision) - 1 if precision > 0 else float('inf')
    r1 = ((1-recall)**2 + os**2)**0.5
    r2 = abs((recall -1 - os)/(2**0.5))
    r_value = 1 - min((r1 + r2) / 2, 1)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 score': f1_score,
        'R-value': r_value,
        'OS': os,
    }

def reduce_with_or(predictions, labels, mask, probs):
    # Reshape tensors to group pairs of indices along the second dimension
    # Add a new dimension, group pairs, and apply the OR operation
    predictions_reduced = predictions.view(predictions.shape[0], -1, 2, 1).max(dim=2).values
    probs_reduced = probs.view(probs.shape[0], -1, 2, 1).max(dim=2).values
    labels_reduced = labels.view(labels.shape[0], -1, 2).max(dim=2).values
    mask_reduced = mask.view(mask.shape[0], -1, 2).max(dim=2).values

    return predictions_reduced, labels_reduced, mask_reduced, probs_reduced

def reduce_sampels_to_20ms(predictions, labels, mask):
    # Reshape tensors to group pairs of indices along the first dimension
    # Add a new dimension, group pairs, and apply the OR operation
    predictions_reduced = predictions.view(-1, 2).max(dim=1).values
    labels_reduced = labels.view(-1, 2).max(dim=1).values
    mask_reduced = mask.view(-1, 2).max(dim=1).values

    return predictions_reduced, labels_reduced, mask_reduced

def calculate_real_model_measurement_20ms(predictions, labels, mask):
    predictions_reduced, labels_reduced, mask_reduced = reduce_with_or(predictions, labels, mask)
    details = calculate_real_model_measurement(predictions_reduced, labels_reduced, mask_reduced)
    return details


def plot_accuracy_curve(results_metric, val=True, train=True, folder=''):
    # Get the directory and filename
    directory = os.path.dirname(folder)
    filename = os.path.basename(folder)

    # Replace 'run_' with 'accuracy_' and change '.log' to '.png' in the filename
    new_filename = filename.replace('run_log', 'accuracy').replace('.log', '.png')

    # Combine the directory and new filename
    accuracy_file_path = os.path.join(directory, new_filename)
    plt.figure(figsize=(10, 5))
    for opt in results_metric:
        try:
            d = results_metric[opt]
            if train:
                training_loss = [i.cpu().numpy().tolist() for i in d['training_acc']]#
                plt.plot(training_loss, label=f'Train accuracy {opt}')
            if val:
                val_acc_history = [i.cpu().numpy().tolist() for i in d['val_acc_history']]#.cpu().numpy().tolist()
                plt.plot(val_acc_history, label=f'Validation accuracy {opt}')
        # plot_results(training_loss,val_acc_history)
        except:
            continue
        
        
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=True)
    plt.savefig(accuracy_file_path)

def plot_loss_curve(results_metric, val=True, train=True, folder=''):
    # Get the directory and filename
    directory = os.path.dirname(folder)
    filename = os.path.basename(folder)

    # Replace 'run_' with 'accuracy_' and change '.log' to '.png' in the filename
    new_filename = filename.replace('run_log', 'loss').replace('.log', '.png')

    # Combine the directory and new filename
    loss_file_path = os.path.join(directory, new_filename)
    plt.figure(figsize=(10, 5))
    for opt in results_metric:
        try:
            d = results_metric[opt]
            
            if train:
            
                training_loss = [i for i in d['training_loss']]#.cpu().numpy().tolist()
                plt.plot(training_loss, label=f'Train Loss {opt}')
            if val:
                val_acc_history = [i for i in d['val_loss']]#.cpu().numpy().tolist()
                plt.plot(val_acc_history, label=f'Validation Loss {opt}')
            
        except:
            continue
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=True)
    plt.savefig(loss_file_path)


def write_sequence_models_statistics_to_file(statistics, file_path):
    filename = os.path.join(file_path, f'statistics_test.txt')
    # Define column width and format for numbers (6 decimal places)
    statistics_0 = statistics[0]
    statistics_0.pop('tolernce', None)
    column_width = 15
    format_string = f"{{:^{column_width}}}"

    # Open the file for writing
    with open(filename, 'w') as file:
        # Write the title
        file.write(f"{' '*(10)}{'-'*(15)} Statistics on the test for 10[msec] frames {'-'*(15)}\n")
        
        # Write the separator line based on the number of keys
        file.write(f"{'-' * 104}\n")
        
        # Create the header row (keys)
        header = " | ".join([format_string.format(key) for key in statistics_0.keys()])
        file.write(header + "\n")

        file.write(f"{'-' * 104}\n")
        
        # Write the separator between the header and the values
        values = " | ".join([format_string.format(f"{value * 100:.2f}%" if key != 'OS' else f"{value:.4f}") for key, value in statistics_0.items()])
        
        file.write(values + "\n")