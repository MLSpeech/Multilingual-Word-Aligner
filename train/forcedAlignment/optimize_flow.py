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

import optuna
import matplotlib.pyplot as plt
from train.forcedAlignment.utils.constants import DEVICE, TRAINING_ARGUMENTS, OUTPUT_OPTUNA_DIR, TIME, FINE_TUNE, DP_PATHES, DATASET, MODEL_NAME
from train.forcedAlignment.train_sequence_model import datasets_for_run
from train.forcedAlignment.utils.train import train_from_scratch
from train.models.utils import initialize_model
from train.models.utils import load_model
from train.forcedAlignment.dynamic_prog.DP_utils import load_DP

NTRAILS = 100
STUDY_NAME = f"optimize_hyperparameters_{TIME}"

def save_optuna_results(study, timestamp):
    """Save Optuna study results and plots in OUTPUT_OPTUNA_DIR."""
    # Create output directory
    if FINE_TUNE:
        output_optuna_dir = os.path.join(OUTPUT_OPTUNA_DIR, f"fine_tuning")
    else:
        output_optuna_dir = OUTPUT_OPTUNA_DIR

    output_dir = os.path.join(output_optuna_dir, f"optimization_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save the study results as a CSV
    csv_path = os.path.join(output_dir, f"{STUDY_NAME}.csv")
    study.trials_dataframe().to_csv(csv_path)
    print(f"Results saved at: {csv_path}")

    # Plot and save contour plot
    # contour_plot = optuna.visualization.plot_contour(study, params=["lr", "alpha", "batch_size"])
    # contour_path = os.path.join(output_dir, f"{STUDY_NAME}_lr_alpha_BatchSize_contour.png")
    # contour_plot.write_image(contour_path)

    # Plot and save contour plot of lr and alpha
    contour_plot = optuna.visualization.plot_contour(study, params=["lr", "alpha"])
    contour_path = os.path.join(output_dir, f"{STUDY_NAME}_lr_alpha_contour.png")
    contour_plot.write_image(contour_path)

    # Plot and save contour plot of gamma and alpha
    contour_plot = optuna.visualization.plot_contour(study, params=["gamma", "alpha"])
    contour_path = os.path.join(output_dir, f"{STUDY_NAME}_gamma_alpha_contour.png")
    contour_plot.write_image(contour_path)

    # Plot and save contour plot of lambbda and alpha
    # contour_plot = optuna.visualization.plot_contour(study, params=["lambbda", "alpha"])
    # contour_path = os.path.join(output_dir, f"{STUDY_NAME}_lambbda_alpha_contour.png")
    # contour_plot.write_image(contour_path)

    # Plot and save parameter importance
    importance_plot = optuna.visualization.plot_param_importances(study)
    importance_path = os.path.join(output_dir, f"{STUDY_NAME}_param_importance.png")
    importance_plot.write_image(importance_path)

    plt.close('all')  # Close plots to avoid memory leaks
    print(f"Plots saved in: {output_dir}")

def optimize_hyperparameters():
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=STUDY_NAME, direction="maximize", sampler=sampler)
    
    #model_arguments = {'model_type':'transformer','sequence_size':80, 'labels_per_input':80,'attention_size':8}
    model_arguments = {'model_type':'conformer','sequence_size':300, 'labels_per_input':300,'number_attention_heads':12, 'conformer_blocks': 16, 'karnel_size': 7}
    #model_arguments = {'model_type':'vgg','vgg_name': 'VGG19N','sequence_size': 31, 'labels_size': 1}
    
    datasets = datasets_for_run(model_arguments)
    if FINE_TUNE:
        # Load the model from the specified path
        dp_config, _, _, _ = load_DP(DP_PATHES[DATASET][MODEL_NAME])
        model_path = dp_config['model_path']
        model_to_train, model_arguments = load_model(model_path)
    else:
        model_to_train = initialize_model(model=model_arguments['model_type'], model_args=model_arguments) #init new model
    
    study.optimize(lambda trial:  train_from_scratch(model_to_train, datasets,
                                                        model_args=model_arguments, device=DEVICE,
                                                        train_cfg=TRAINING_ARGUMENTS, optuna=True, trial=trial),
                                                        n_trials=NTRAILS) #trial.report  # , timeout=16*60


    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")

    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Save results and plots
    save_optuna_results(study, TIME)
    
if __name__ == '__main__':
    optimize_hyperparameters()