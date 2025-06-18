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
import shutil
from train.evaluation.compare_results import calculate_real_model_measurement
from train.forcedAlignment.utils.constants import DEVICE, TRAINING_ARGUMENTS, TIME, FINAL_TRAIN, RUN_EXPERIMENTS, FINE_TUNE, LABELS_DIR_VAL, DP_PATHES, DATASET, MODEL_NAME
import train.forcedAlignment.utils.custom_loss as loss_class
import time
import copy
from train.forcedAlignment.utils.logger_utils import setup_logging, log_details, setup_logging_fine_tuning
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from train.forcedAlignment.dynamic_prog.train_DP import update_measures_by_validation_evaluation
from train.forcedAlignment.dynamic_prog.DP_utils import load_DP
from train.forcedAlignment.utils.preprocess import get_all_model_files_names
from train.forcedAlignment.dynamic_prog.dynamic_prog_statistics import calculate_statistics, log_best_statistics_fine_tune

def set_parameter_requires_grad(model, feature_extracting=True):
    # approach 1
    if feature_extracting:
        # frozen model
        model.requires_grad_(False)
    else:
        # fine-tuning
        model.requires_grad_(True)


    
def save_model(model, output_dir, model_config, log_file_path):
    timestamp = TIME
    if FINE_TUNE:
        run_dir = os.path.dirname(log_file_path)
    else:
        # Create a directory for this run
        model_name = model_config['vgg_name'] if 'vgg_name' in model_config else model_config['model_type']
        sequence_size = model_config['sequence_size'] if 'sequence_size' in model_config else ''

        if model_config['model_type'] == 'vgg':
            run_dir = os.path.join(output_dir, f'{model_name}_seq{sequence_size}_{timestamp}')
        elif model_config['model_type'] == 'transformer':
            run_dir = os.path.join(output_dir, f"{model_name}_seq{sequence_size}_att{model_config['attention_size']}_{timestamp}")
        elif model_config['model_type'] == 'conformer':
            run_dir = os.path.join(output_dir, f"{model_name}__seq{sequence_size}_head{model_config['number_attention_heads']}_conformer_blocks{model_config['conformer_blocks']}_kernel{model_config['karnel_size']}_{timestamp}")
        
        os.makedirs(run_dir, exist_ok=True)

    # Save the model's state_dict inside the run directory
    model_save_path = os.path.join(run_dir, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_save_path)
    
    # Save the model's configuration as well
    config_save_path = os.path.join(run_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(model_config, f, indent=4)

    if RUN_EXPERIMENTS and (not FINE_TUNE):
        shutil.copy(log_file_path, run_dir)

    print(f"Model and configuration saved at {run_dir}")
    print("Timestamp: ", timestamp)

def calculate_running_corrects(preds, labels, masks):
    # Flatten tensors
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    masks_flat = masks.view(-1).bool()  # Ensure mask is boolean

    # Apply the mask to filter out unwanted elements
    filtered_preds = preds_flat[masks_flat]
    filtered_labels = labels_flat[masks_flat]

    # Calculate the number of correct predictions
    running_corrects = torch.sum(filtered_preds == filtered_labels) #.item()

    total_labels = masks_flat.sum() #.item()

    return running_corrects, total_labels




def train_cfg_for_optuna(trial, train_cfg=TRAINING_ARGUMENTS):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # log=True, will use log scale to interplolate between lr
    alpha = trial.suggest_float("alpha", 0.68, 0.90, step=0.01)
    gamma = trial.suggest_float("gamma", 1.1, 1.9, step=0.01)
    #lambbda = trial.suggest_float("lambbda", 0.4, 1, step=0.01)
    batch_size = 16 #trial.suggest_categorical("batch_size", [16])

    train_cfg['learning_rate'] = lr
    train_cfg['alpha'] = alpha
    train_cfg['gamma'] = gamma
    #train_cfg['lambbda'] = lambbda
    train_cfg['num_epochs']= 8
    train_cfg['batch_size'] = batch_size
    return train_cfg
    

def train_from_scratch(model, datasets, model_args=None, device=DEVICE, train_cfg=TRAINING_ARGUMENTS, optuna=False, trial=None):

    if optuna:
        train_cfg = train_cfg_for_optuna(trial, train_cfg)

    dp_cfg = None
    learning_rate_coefficient = 1
    if FINE_TUNE:
        dp_cfg, w, features_object = load_DP(DP_PATHES[DATASET][MODEL_NAME])
        labels_files_val = get_all_model_files_names(labels_dir=LABELS_DIR_VAL)
        best_accuracy_0_dp = 0
        best_statistics_dp = {}
        learning_rate_coefficient=0.01

    batch_size = train_cfg['batch_size']
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    dataloaders = {'train':train_loader, 'val': val_loader}
    
    optimizer = Adam(model.parameters(), lr=train_cfg['learning_rate']*learning_rate_coefficient)
    
    if FINE_TUNE:
        logger, log_file_path = setup_logging_fine_tuning(dp_cfg, TIME, model_args, train_cfg, run_exp=RUN_EXPERIMENTS, optuna=optuna)
    else:
        logger, log_file_path = setup_logging(train_cfg, TIME, model_args, optuna=optuna, run_exp=RUN_EXPERIMENTS)

    since = time.time()
    num_epochs = train_cfg['num_epochs']
    val_acc_history = []
    training_acc = []
    training_loss = []
    val_loss = []
    loss_criteria = loss_class.BinaryFocalLoss(alpha=train_cfg['alpha'], gamma=train_cfg['gamma'], lambbda=train_cfg['lambbda'], reduction='mean', tolerance_window=3, penalized=False, regularization=FINE_TUNE)
    model = model.to(device)
    best_val_details = 0
    best_details = {}
    epochs_time = []
    
    for epoch in range(num_epochs):
        start_epoch = time.time()
        logger.warning('Epoch {}/{}'.format(epoch+1, num_epochs))
        logger.warning('-' * 10)

        all_preds = []
        all_labels = []
        all_masks = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_labels = 0

            # Iterate over data.
            for inputs, labels, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = loss_criteria(outputs, labels, masks, model_args['model_type'])
                    # Apply sigmoid to logits to get probabilities
                    probabilities = torch.sigmoid(outputs)
                    
                    if model_args['model_type'] != 'vgg':
                        probabilities = probabilities.squeeze(-1) # Shape: [64, 200]

                    # Convert probabilities to binary predictions (0 or 1)
                    preds = (probabilities > 0.5).float()  # Shape: [64, 200]
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if phase == 'val' and ((epoch%10 == 9) or (epoch == num_epochs - 1) or optuna or FINAL_TRAIN):
                    all_preds.append(preds.detach())
                    all_labels.append(labels.detach())
                    all_masks.append(masks.detach())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                epoch_running_corrects, epoch_num_labels = calculate_running_corrects(preds, labels.data, masks)
                running_corrects += epoch_running_corrects
                total_labels += epoch_num_labels
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset) # we have batch*length labels - loss is mean over batch and length
            epoch_acc = running_corrects.double() / total_labels.double()
            logger.warning('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and ((epoch%10 == 9) or (epoch == num_epochs - 1) or optuna or FINAL_TRAIN):
                all_preds = torch.cat(all_preds, dim=0) #torch.Size([966, 300])
                all_labels = torch.cat(all_labels, dim=0) #torch.Size([966, 300])
                all_masks = torch.cat(all_masks, dim=0) #torch.Size([966, 300])
                details = calculate_real_model_measurement(all_preds, all_labels, all_masks) #os, recall, precision, accuracy 
                log_details(details, logger)

                if FINE_TUNE:
                    measures = update_measures_by_validation_evaluation(labels_files_val, model_args, model, dp_cfg, w, features_object)
                    all_statistics = calculate_statistics(measures, [0,2,4,9], logger)
                    if all_statistics[0]['Accuracy'] > best_accuracy_0_dp:
                        best_accuracy_0_dp = all_statistics[0]['Accuracy']
                        best_statistics_dp = all_statistics
                        best_details = details
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if optuna:
                        trial.report(all_statistics[0]['Accuracy'], epoch)

                else: 
                    if details['F1 score']>=best_val_details:
                        best_val_details = details['F1 score'] 
                        best_details = details
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if optuna:
                        trial.report(details['F1 score'], epoch)
                        
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)
            if phase == 'train':
                training_acc.append(epoch_acc)
                training_loss.append(epoch_loss)
        
        epochs_time.append(time.time() - start_epoch)
        logger.warning('Epoch complete in {:.0f}h {:.0f}m {:.0f}s'.format((time.time() - start_epoch) // 3600, ((time.time() - start_epoch) % 3600) // 60, (time.time() - start_epoch) % 60))
        
    # logging the loss of the train and val
    if FINE_TUNE:
        log_best_statistics_fine_tune(best_statistics_dp, logger)
        logger.warning(f"\n------  The best Statistics Base on Fine Tuning:  ------")
    else:
        logger.warning("------  The best Statistics  ------")
    
    log_details(best_details, logger)
    logger.warning(f"train loss: {training_loss}")
    logger.warning(f"validation loss: {val_loss}")
    time_elapsed = time.time() - since
    logger.warning('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    best_details['avg_epoch_time'] = np.array(epochs_time).mean()
    model.load_state_dict(best_model_wts)
    
    if optuna:
        if FINE_TUNE:
            print("best details: ", best_statistics_dp)
            return best_accuracy_0_dp
        else:
            print("best details: ", best_details)
            return best_val_details
    else:
        return model, training_acc, val_acc_history, training_loss, val_loss, best_details, log_file_path