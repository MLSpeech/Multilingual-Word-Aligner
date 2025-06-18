import os
import sys
import gc
import pandas as pd

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
from train.forcedAlignment.utils.train import train_from_scratch, save_model
from train.models.utils import initialize_model, load_model
from train.forcedAlignment.utils.preprocess import prepare_dataset, EmbeddedDataset
import train.forcedAlignment.utils.constants as constants
from train.evaluation.compare_results import plot_accuracy_curve,plot_loss_curve
from train.forcedAlignment.dynamic_prog.DP_utils import load_DP


def datasets_for_run(model_arguments=None):
    embeddings, labels, masks = prepare_dataset(labels_dir=constants.LABELS_DIR, mode='train', model_arguments=model_arguments)
    embeddings_val, labels_val, masks_val = prepare_dataset(labels_dir=constants.LABELS_DIR_VAL, mode='val', model_arguments=model_arguments) 

    train_dataset = EmbeddedDataset(embeddings, labels, masks)
    validation_set = EmbeddedDataset(embeddings_val, labels_val, masks_val)
    return {'train':train_dataset, 'val':validation_set}
    
def train_eval_model(model_arguments=None):
    print("device is: ", constants.DEVICE)

    # Load an existing model for fine-tuning (if FINE_TUNE is enabled)
    if constants.FINE_TUNE:
        # Load the model from the specified path
        dp_config, _, _ = load_DP(constants.DP_PATHES[constants.DATASET][constants.MODEL_NAME])
        model_path = dp_config['model_path']
        model_to_train, model_arguments = load_model(model_path)
        print(f"Fine-tuning model loaded from: {model_path}")
    else:
        model_to_train = initialize_model(model=model_arguments['model_type'], model_args=model_arguments) #init new model
        print("New model initialized.")
    
    datasets = datasets_for_run(model_arguments)
    try:
        # Use DataLoader to create batches and shuffle the training data
        model, training_acc, val_acc_history, training_loss, val_loss, best_details, log_file_path  = train_from_scratch(model_to_train, datasets, model_arguments, device=constants.DEVICE) 
        results = {'1':{'training_acc':training_acc,'val_acc_history':val_acc_history,'training_loss':training_loss,'val_loss':val_loss}}
        plot_accuracy_curve(results, folder=log_file_path)
        plot_loss_curve(results, folder=log_file_path)
    
    finally:
        # Free GPU memory
        del model_to_train
        gc.collect()
        torch.cuda.empty_cache()
        print("Freed GPU memory")
    
    if constants.SAVE_MODEL:
        save_model(model, constants.OUTPUT_DIR, model_arguments, log_file_path)
        
    return best_details

def main(model=None, sequence_sizes=None):
    model = constants.MODEL_NAME
    results = []

    if model == 'Transformer':
        ####   Transformer    ###
        sequence_sizes = [80]
        attention_sizes = [8]
        for seq_size in sequence_sizes:
            for attention_size in attention_sizes:
                model_arguments = {'model_type':'transformer','sequence_size':seq_size, 'labels_per_input':seq_size,'attention_size':attention_size}
                result_iteration = train_eval_model(model_arguments=model_arguments)
                results.append([seq_size, attention_size, result_iteration])

    elif model == 'Conformer':
        ##   Conformer   ###
        sequence_sizes = [300]
        number_of_attention_heads = [12]
        conformer_blocks = [16]
        karnel_sizes = [7]
        for conformer_block in conformer_blocks:
            for karnel_size in karnel_sizes:
                for seq_size in sequence_sizes:
                    for head in number_of_attention_heads: 
                        model_arguments = {'model_type':'conformer','sequence_size':seq_size, 'labels_per_input':seq_size,'number_attention_heads':head, 
                                        'conformer_blocks': conformer_block, 'karnel_size': karnel_size}
                        result_iteration = train_eval_model(model_arguments=model_arguments)
                        results.append([seq_size, head, result_iteration])

    elif model == 'VGG':
        ###   VGG   ###
        models = ['VGG19N']
        sequence_sizes = [31]
        for model_type in models:
            for size in sequence_sizes: 
                model_arguments = {'model_type':'vgg','vgg_name': model_type,'sequence_size': size, 'labels_size': 1}
                result_iteration = train_eval_model(model_arguments=model_arguments)
                results.append([model_type, size, result_iteration])


    if constants.RUN_EXPERIMENTS and not constants.FINE_TUNE:
        model_measurements = pd.DataFrame(results)
        # Check if file exists to handle headers
        file_exists = os.path.isfile(f'{constants.OUTPUT_LOG_DIR}/results.csv')
        
        model_measurements.to_csv(f'{constants.OUTPUT_LOG_DIR}/results.csv', 
                                index=False, 
                                mode='a', 
                                header=not file_exists)  # Write header only if file doesn't exist


if __name__ == "__main__":
    main()