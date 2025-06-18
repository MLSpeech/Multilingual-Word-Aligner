import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

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


from train.forcedAlignment.utils.preprocess import prepare_dataset, get_all_model_files_names
from train.forcedAlignment.utils.constants import DEVICE, VAL_FILES, TEST_FILES, MODEL_NAME, MODEL_PATHS, DATASET, DP_PATHES, MODEL_TRAINED_DATASET, TIMIT, BUCKEYE
from train.models.utils import load_model
from train.evaluation.compare_results import calculate_real_model_measurement, reduce_with_or, score_boundaries, write_sequence_models_statistics_to_file
from train.forcedAlignment.dynamic_prog.train_DP import update_measures_by_validation_evaluation
from train.forcedAlignment.dynamic_prog.DP_utils import load_DP, write_statistics_to_file
from train.forcedAlignment.dynamic_prog.dynamic_prog_statistics import calculate_statistics
from train.forcedAlignment.utils.logger_utils import log_details

TEST = True
VAL = False
TWENTY_MS = False
DP = True

# Check if only one is True
if sum([TEST, VAL]) > 2:
    raise ValueError("Exactly one or zero constant must be True, but this condition is not met!")

def save_heatmap(data, path):
    """
    Generate and save a heatmap from a 2D list rotated by 90 degrees.
    
    Parameters:
    - data: 2D list (list of lists)
    - path: File path to save the heatmap PNG
    """
    # Convert the 2D list to a NumPy array
    data_array = data

    # Rotate the data by transposing it
    rotated_data = np.transpose(data_array)

    # Create the heatmap
    plt.figure(figsize=(10, 8))  # Optional: adjust the figure size
    plt.imshow(rotated_data, cmap='viridis', aspect='auto')
    plt.colorbar()
    #plt.show()

    # Save the heatmap as a PNG file
    plt.savefig(path, format='png')

    # Close the plot to free up memory
    plt.close()

def select_random_files(directory_path):
    # Dictionary to hold files for each prefix
    files_by_prefix = {
        "dr5_": [],
        "dr6_": [],
        "dr7_": [],
        "dr8_": []
    }

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file ends with '.wrd' and starts with the expected prefixes
        if filename.endswith('.wrd'):
            for prefix in files_by_prefix:
                if filename.startswith(prefix):
                    files_by_prefix[prefix].append(filename)

    # Randomly select one file from each prefix
    selected_files = []
    for prefix, files in files_by_prefix.items():
        if files:
            selected_files.append(random.choice(files))
        else:
            print(f"No files found for prefix {prefix}")
            return None
    
    return selected_files

def create_tensor_from_indices(y, masks, conf_model):
    if conf_model['model_type'] == 'vgg':
        tensor = torch.zeros((len(masks), 1, 1), dtype=torch.float32)
        for index in y:
            tensor[index][0][0] = 1
    else:
        tensor = torch.zeros((len(masks), conf_model['sequence_size'], 1), dtype=torch.float32)
        # Count the number of False values
        false_count = np.count_nonzero(masks[-1] == False)
        # Set the corresponding indices in the tensor to 1
        for index in y:
            # Calculate the row and column based on the index in the list
            row = index // conf_model['sequence_size']  # Row is index divided by 300 (integer division)
            col = index % conf_model['sequence_size']   # Column is the remainder of the division
            if (row == (len(masks) - 1)) and (len(masks) != 1):
                col = col + false_count
            
            # Set the value at the specific index to 1
            tensor[row][col][0] = 1

    return tensor


def get_predictions(model, eval_embedding, device):
    """
    Get model predictions for the given dataset.

    Parameters:
    - model: The trained model.
    - dataloader: A DataLoader instance that provides batches of input data.
    - device: The device to which the model and data are moved.

    Returns:
    - predictions: A numpy array containing the model's predictions.
    """
    model.eval()
    all_outputs_predictions = []
    all_outputs_probabilities = []

    with torch.no_grad():
        for batch in eval_embedding:
            input_ids = torch.tensor(np.array([batch]), dtype=torch.float32).to(device)
            outputs= model(input_ids=input_ids)
            # Apply sigmoid activation to get probabilities if needed
            logits = outputs
            probabilities = torch.sigmoid(logits)  # Apply sigmoid if your task requires probabilities
            predictions = (probabilities >= 0.5).float()
            all_outputs_probabilities.append(probabilities)
            all_outputs_predictions.append(predictions)
            

    # Concatenate all batches of outputs into a single numpy array
    predictions = torch.cat(all_outputs_predictions, dim=0)
    probabilities = torch.cat(all_outputs_probabilities, dim=0)
    return predictions, probabilities


def find_corresponding_wav(wrd_filename, directory_path=VAL_FILES):
    # Ensure the file ends with .wrd
    if not wrd_filename.endswith('.wrd'):
        raise ValueError("The input file must end with '.wrd'")

    # Get the base name of the file (without extension)
    base_name = os.path.splitext(wrd_filename)[0]

    # Create the expected .wav filename
    wav_filename = base_name + '.wav'

    # Get the full path of the expected .wav file
    wav_file_path = os.path.join(directory_path, wav_filename)

    # Check if the .wav file exists in the directory
    if os.path.exists(wav_file_path):
        return wav_file_path
    else:
        raise FileNotFoundError(f"The corresponding .wav file '{wav_filename}' was not found in the directory")
    

def sequence_models_evaluation(labels_files, model, device, conf_model, model_type, files_folder, model_path):
    measures = {
        0: {'TP':0,'TP+FP':0,'TP+FN':0},
        2: {'TP':0,'TP+FP':0,'TP+FN':0},
        4: {'TP':0,'TP+FP':0,'TP+FN':0},
        9: {'TP':0,'TP+FP':0,'TP+FN':0}
    }
        
    all_preds = []
    all_labels = []
    all_masks = []
    
    for file in labels_files:
        # Prepare evaluation data
        if TEST:
            embeddings_val, labels_val, masks_val = prepare_dataset(labels_dir=TEST_FILES, mode='test', model_arguments=conf_model, specific_file=file)
        else:    
            embeddings_val, labels_val, masks_val = prepare_dataset(labels_dir=VAL_FILES, mode='val', model_arguments=conf_model, specific_file=file)
        
        if not embeddings_val:
            continue
        
        # Get predictions
        predictions, probs = get_predictions(model, embeddings_val, device) #Size([2, 300, 1])

        labels_val = torch.tensor(np.array(labels_val), dtype=torch.float32).to(device) #Size([2, 300])
        masks_val = torch.tensor(np.array(masks_val), dtype=torch.float32).to(device) #Size([2, 300])

        
        if TWENTY_MS:
            if model_type == 'VGG':
                predictions = predictions.unsqueeze(0)
                probs = probs.unsqueeze(0)
                labels_val = labels_val.transpose(0, 1)
                masks_val = masks_val.transpose(0, 1)

            predictions, labels_val, masks_val, probs = reduce_with_or(predictions, labels_val, masks_val, probs)

            if model_type == 'VGG':
                predictions = predictions.squeeze(0)
                probs = probs.squeeze(0)
                labels_val = labels_val.transpose(0, 1)
                masks_val = masks_val.transpose(0, 1)
        
        all_preds.append(predictions.detach())
        all_labels.append(labels_val.detach())
        all_masks.append(masks_val.detach())
        predictions = predictions.view(-1) #predictions shape:  torch.Size([600])
        labels_val = labels_val.view(-1) #labels_val shape:  torch.Size([600])
        masks_val = masks_val.view(-1) #masks_val shape:  torch.Size([600])
        predictions = predictions[masks_val.bool()] #predictions shape:  torch.Size([402])
        labels_val = labels_val[masks_val.bool()] #labels_val shape:  torch.Size([402])
        for tol in [0,2,4,9]:
            tp, tp_plus_fp, tp_plus_fn = score_boundaries([labels_val.cpu().numpy()], [predictions.cpu().numpy()], tol)
            measures[tol]['TP'] += tp
            measures[tol]['TP+FP'] += tp_plus_fp
            measures[tol]['TP+FN'] += tp_plus_fn

  
    ##   Statistics   ##
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    details = calculate_real_model_measurement(all_preds, all_labels, all_masks)
    
    print(f"\n  {' '*(18)} Statistics")
    statistics_windows = []
    for tol in [0,2,4,9]:
        accuracy = details['Accuracy']
        precision = measures[tol]['TP'] / measures[tol]['TP+FP'] if measures[tol]['TP+FP'] > 0 else 0
        recall = measures[tol]['TP'] / measures[tol]['TP+FN'] if measures[tol]['TP+FN'] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        os = (recall / precision) - 1 if precision > 0 else float('inf')
        r1 = ((1-recall)**2 + os**2)**0.5
        r2 = abs((recall -1 - os)/(2**0.5))
        r_value = 1 - min((r1 + r2) / 2, 1)
        statistics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 score': f1_score,
            'R-value': r_value,
            'OS': os,
        }
        window = (tol + 1)*10
        num_of_dash = 104
        print(f"{'-'*(num_of_dash)}\n {' '*(15)} Tolerance: {window} [msec]")
        print("-"*(num_of_dash))
        log_details(statistics, None, print_to_log=False)
        statistics['tolernce'] = tol
        statistics_windows.append(statistics)
    
    write_sequence_models_statistics_to_file(statistics_windows, model_path)
        
    return details, statistics_windows
    


def main(model_path, name_mapping=None):
    print(f"{'-'*(79 - 4)}\n started model :{model_path}")
    print("-"*(79 - 4))
    # Load the trained model
    if 'vgg' in model_path.lower():
       model_type = 'VGG'
    elif 'transformer' in model_path.lower():
       model_type = 'transformer'
    elif 'conformer' in model_path.lower():
       model_type = 'conformer'
    else:
        raise Exception(f'no known model for path {model_path}')
           
    if DP:
        dp_cfg, w, features_object = load_DP(DP_PATHES[MODEL_TRAINED_DATASET][MODEL_NAME])
        model_path = dp_cfg['model_path']
        
    device = DEVICE
    model, conf_model = load_model(model_path, device)

    if TEST:
        labels_files = get_all_model_files_names(labels_dir=TEST_FILES)
        files_folder = TEST_FILES
        mode = 'test'
    elif VAL:
        labels_files = get_all_model_files_names(labels_dir=VAL_FILES)
        files_folder = VAL_FILES
        mode = 'val'
    else:
        labels_files = select_random_files(directory_path=files_folder)
        mode = 'val'

    if DP:
        eval_tol = [0, 0.5, 1, 1.5, 4, 9] #10=0 15=0.5 20=1 25=1.5 50=4 100=9
        measures = update_measures_by_validation_evaluation(labels_files, conf_model, model, dp_cfg, w, features_object, mode=mode, tol_lst=eval_tol, labels_dir=files_folder)
        all_statistics = calculate_statistics(measures, eval_tol, None, print_to_log=False, test=TEST)
        if DATASET == TIMIT or DATASET == BUCKEYE:
            write_statistics_to_file(all_statistics, DP_PATHES[DATASET][MODEL_NAME], TEST, DATASET)
    else:
        details, statistics_windows = sequence_models_evaluation(labels_files, model, device, conf_model, model_type, files_folder, model_path)

    print(f"{'-'*(79 - 4)}\n finished model :{model_type}")
    print("-"*(79 - 4))   


if __name__ == "__main__":
    #analyze_and_latex()
    model_path = MODEL_PATHS[MODEL_TRAINED_DATASET][MODEL_NAME]['early_stop']
    main(model_path)