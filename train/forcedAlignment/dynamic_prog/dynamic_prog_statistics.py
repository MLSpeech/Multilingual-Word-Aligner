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

from train.forcedAlignment.utils.logger_utils import log_details
import wandb




def calculate_statistics(measures, tols, logger, print_to_log=True, test=False):
    all_statistics = {}

    for tol in tols:
        # Calculate accuracy
        accuracy = measures[tol]['TP'] / measures[tol]['TP+FP'] if measures[tol]['TP+FP'] > 0 else 0
        # Calculate OS (Oversegmentation)
        os = (accuracy / accuracy) - 1 if accuracy > 0 else float('inf')
        # Calculate R-value
        r1 = ((1 - accuracy)**2 + os**2)**0.5
        r2 = abs((accuracy - 1 - os) / (2**0.5))
        r_value = 1 - min((r1 + r2) / 2, 1)

        # Store statistics
        statistics = {
            'Accuracy': accuracy,
            'R-value': r_value,
            'OS': os,
        }
        # Remove the 'OS' key if test is True
        if test:
            statistics.pop('OS', None)

        # Log the statistics for this tolerance value
        window = (tol + 1)*10
        if print_to_log:
            logger.warning(f"{'-'*(60)}\n {' '*(15)} Tolerance: {window} [msec]")
            logger.warning("-"*(60))
        else:
            print(f"{'-'*(60)}\n {' '*(15)} Tolerance: {window} [msec]")
            print("-"*(60))
        log_details(statistics, logger, print_to_log=print_to_log)

        # Save statistics in all_statistics dictionary
        all_statistics[tol] = statistics

    return all_statistics



def log_best_statistics_fine_tune(best_statistics, logger):
    logger.warning(f"\n {' '*(1)} Best Statistics Measure By Dynamic Programing:")
    for tol in [0, 2, 4, 9]:
        window = (tol + 1)*10
        logger.warning(f"{'-'*(60)}\n {' '*(10)} Tolerance: {window} [msec]")
        logger.warning("-"*(60))
        log_details(best_statistics[tol], logger)


def log_best_statistics(best_statistics, logger):
    table = wandb.Table(columns=["Tolerance [msec]", "Accuracy [%]", "R-value [%]", "OS [%]"])
    logger.warning("Best statistics are:")
    
    # Loop through the list of tolerance values
    for tol in [0, 2, 4, 9]:
        window = (tol + 1)*10
        logger.warning(f"{'-'*(60)}\n {' '*(15)} Tolerance: {window} [msec]")
        logger.warning("-"*(60))
        log_details(best_statistics[tol], logger)
        table.add_data(window, best_statistics[tol]['Accuracy']*100, best_statistics[tol]['R-value']*100, best_statistics[tol]['OS']*100)

    wandb.log({"tolerance_table": table})
