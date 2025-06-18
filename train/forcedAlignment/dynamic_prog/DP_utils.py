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
import json
import wandb
import numpy as np
import torch
from train.forcedAlignment.dynamic_prog.extract_features import load_token_statistics, Features_DP

def wandb_log_statistics(all_statistics, batch_num):

    for key in [0, 2, 4, 9]:
        # Ensure the key exists in the dictionaries
        if key in all_statistics:
            wandb.log({f"accuracy_tol_{key}": float(all_statistics[key]["Accuracy"]), "batch_num": batch_num})
            wandb.log({f"r_value_tol_{key}": float(all_statistics[key]["R-value"]), "batch_num": batch_num})


def save_config_and_weights(save_dir, train_config, final_weights):
    # Convert NumPy arrays to lists before saving
    final_weights_list = final_weights.tolist()

    # Add final_weights to the train_config dictionary
    train_config['final_weights'] = final_weights_list

    # Save the updated train_config dictionary to a config.json file
    config_path = f"{save_dir}/config.json"
    with open(config_path, 'w') as f:
        json.dump(train_config, f, indent=4)
    print(f"Config and weights saved to {config_path}")


def load_config_and_weights(config_path):
    # Load the config data from the JSON file
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    
    # Extract the final_weights and convert from list back to Torch tensor
    final_weights = torch.from_numpy(np.array(train_config['final_weights']))

    return train_config, final_weights


# Helper function to chunk a list into sub-lists
def create_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def create_tensor_from_indices_masked(y, labels):
    tensor = torch.zeros(len(labels), dtype=torch.float32)
    for index in y:
        tensor[index] = 1

    return tensor

def load_DP(dp_path):
    config_path = os.path.join(dp_path, f'config.json')
    dp_cfg, w = load_config_and_weights(config_path)
    features_object = Features_DP(dp_cfg['features'])
    
    return dp_cfg, w, features_object


def write_statistics_to_file(statistics, file_path, test, dataset):
    if test:
        stat_name = f'{dataset}_statistics_test.txt'
    else:
        stat_name = '{dataset}_statistics_val.txt'
    filename = os.path.join(file_path, stat_name)
    # Calculate the number of keys in the statistics dictionary
    num_of_keys_in_statistics_dict = len(statistics)

    # Set the column width and format string
    column_width = 17
    format_string = f"{{:^{column_width}}}"

    # Open the file for writing
    with open(filename, 'w') as file:
        # Write the title
        file.write(f"--- Alignment Accuracy on the test [%] ---\n")
        
        # Write the separator line based on the number of keys
        file.write(f"{'-' * (20 * num_of_keys_in_statistics_dict)}\n")
        
        # Create the header row (keys)
        header = " | ".join([format_string.format(f"t ≤ {(key+1)*10}[msec]") for key in statistics.keys()])
        file.write(header + "\n")
        
        # Write the separator between the header and the values
        file.write(f"{'-' * (20 * num_of_keys_in_statistics_dict)}\n")
        
        # Write the values for the "Accuracy" key from each inner dictionary
        accuracy_values = " | ".join([format_string.format(f"{statistics[key]['Accuracy'] * 100:.2f}") for key in statistics.keys()])
        file.write(accuracy_values + "\n")