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
from pathlib import Path
from forcedAlignment.utils.constants import models_configurations, TEST_FILES
from forcedAlignment.utils.preprocess import get_all_model_files_names, get_labels_in_sec
from evaluation.compare_results import score_boundaries_DP
from forcedAlignment.dynamic_prog.dynamic_prog_statistics import calculate_statistics


def prepare_mms_pred(mms_file: str, files_folder):
    mms_file = os.path.join(files_folder,f'{mms_file}_mms.txt')
    end_times = []
    transcript_started = False

    # Read the file line by line
    with open(mms_file, 'r') as file:
        for line in file:
                line = line.strip()
                
                if not transcript_started:
                    if line.startswith("Transcript:"):
                        transcript_started = True
                    continue
                
                if line.startswith("Words spans:"):
                    continue
                    #break
                
                parts = line.split()
                if len(parts) == 5:
                    end_time, score = float(parts[3][:-1]), float(parts[4])
                    end_times.append(end_time)

    return end_times





def main():
    #dirs_lst = ["train", "test", "val"]

    mode='test'
    if mode == 'train':
        configuration_folder_name = 'train_folder'
    elif mode == 'val':
        configuration_folder_name = 'val_folder'
    elif mode == 'test':
        configuration_folder_name = 'test_folder'
    else:
        raise Exception("enter a valid mode")


    labels_files = get_all_model_files_names(labels_dir=TEST_FILES)
    eval_tol = [0, 0.5, 1, 1.5, 4, 9] #10=0 15=0.5 20=1 25=1.5 50=4 100=9
    measures = {tol: {'TP': 0, 'TP+FP': 0, 'TP+FN': 0} for tol in eval_tol}

    for file in labels_files:
        try:
            file_name = Path(file).stem
            labels_in_sec = get_labels_in_sec(TEST_FILES, file)
            mms_pred = prepare_mms_pred(file_name, models_configurations['mms'][configuration_folder_name])

            for tol in eval_tol:
                tp, tp_plus_fp, tp_plus_fn = score_boundaries_DP(labels_in_sec, mms_pred, tol, scale=0.02)
                measures[tol]['TP'] += tp
                measures[tol]['TP+FP'] += tp_plus_fp
                measures[tol]['TP+FN'] += tp_plus_fn

        except FileNotFoundError:
            print(f"File '{file_name}' does not exist. Skipping...")
            continue

    
    _ = calculate_statistics(measures, eval_tol, None, print_to_log=False, test=True)


if __name__ == "__main__":
    main()