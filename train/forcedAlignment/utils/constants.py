import torch
import os
import sys
import time

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

from train.forcedAlignment.utils.gpu_selector import get_best_gpu
TIMIT = 'timit'
BUCKEYE = 'buckeye'
HEBREW = 'hebrew'
DUTCH = 'IFACorpus' #'dutch'
GERMAN = 'phonedat' #'german'
DATASET = TIMIT


MODEL_TRAINED_DATASET = TIMIT

# NOTE:
# Before using this script, make sure to set the variable `PATH_TO_DATA_DIR`
# to the correct location of your local dataset directory.
# This path may vary depending on your team's storage setup.
# Example:
# PATH_TO_DATA_DIR = "/your/local/path/to/data"
PATH_TO_DATA_DIR = f''

DATASETS_MAPPING = {
    TIMIT:{
        'mms_repeat': 64, # 64 for unsup - 60 for Dseg -> 68 for mms  #68 - Exp no tzvia
        'DsegkNN_repeat': 3,
        'UnSupSeg_size': 64,
        'conformer_fc_size': 128, #192 - Exp no tzvia
        'transformer_size': 128, #192 - Exp no tzvia
        'embeddings_size': 128, #192 - Exp no tzvia
        'word_posix': '.wrd'
    },
    BUCKEYE:{
        'mms_repeat': 128,#128 + 128 - 120,  #136 - Exp no tzvia
        'DsegkNN_repeat': 6, #6 * 20 (size of Dseg output) = 120
        'UnSupSeg_size': 128,
        'conformer_fc_size': 256,
        'transformer_size': 256,
        'embeddings_size': 256,
        'word_posix': '.word'
    },
    HEBREW:{
        'mms_repeat': 128, # 64 for unsup - 60 for Dseg -> 68 for mms  #68 - Exp no tzvia
        'DsegkNN_repeat': 3,
        'UnSupSeg_size': 128,
        'conformer_fc_size': 256, #192 - Exp no tzvia
        'transformer_size': 256, #192 - Exp no tzvia
        'embeddings_size': 256, #192 - Exp no tzvia
        'word_posix': '.wrd'
    },
    DUTCH: {
        'mms_repeat': 128,#128 + 128 - 120,  #136 - Exp no tzvia
        'DsegkNN_repeat': 6, #6 * 20 (size of Dseg output) = 120
        'UnSupSeg_size': 128,
        'conformer_fc_size': 256,
        'transformer_size': 256,
        'embeddings_size': 256,
        'word_posix': '.wrd'
    },
    GERMAN: {
        'mms_repeat': 64, # 64 for unsup - 60 for Dseg -> 68 for mms  #68 - Exp no tzvia
        'DsegkNN_repeat': 3,
        'UnSupSeg_size': 64,
        'conformer_fc_size': 128, #192 - Exp no tzvia
        'transformer_size': 128, #192 - Exp no tzvia
        'embeddings_size': 128, #192 - Exp no tzvia
        'word_posix': '.wrd'
    },
}

MODEL_PATHS = {
    TIMIT: {
        'Conformer': {
            'early_stop': '',
            'no_early_stop': ''
        },
        'Transformer': {
            'early_stop': '',
            'no_early_stop': ''
        },
        'VGG':{
            'early_stop': '',
            'no_early_stop': ''            
        }
    },
    BUCKEYE:{
        'Conformer': {
            'early_stop': '',
            'no_early_stop': ''
        },
        'Transformer': {
            'early_stop': '',
            'no_early_stop': ''
        },
        'VGG':{
            'early_stop': '',
            'no_early_stop': ''            
        }
    }
}



DP_PATHES = {
    TIMIT: {'Conformer': '',
            'Transformer': '',
            'VGG': ''},
    BUCKEYE: {'Conformer': '',
            'Transformer': '',
            'VGG': ''}
}



# Define the input directory where your files of the deteset are located
LABELS_DIR = os.path.join(PATH_TO_DATA_DIR, f'{DATASET}/train/')
TRAIN_FILES = os.path.join(PATH_TO_DATA_DIR, f'{DATASET}/train/')
VAL_FILES =  os.path.join(PATH_TO_DATA_DIR, f'{DATASET}/val/') 
LABELS_DIR_VAL =  os.path.join(PATH_TO_DATA_DIR, f'{DATASET}/val/') 
TEST_FILES =  os.path.join(PATH_TO_DATA_DIR, f'{DATASET}/test/') 
MMS_EMISSIONS_FOLDER =  os.path.join(project_dir, f'train/models/MMS/mms_emissions/{DATASET}')



best_gpu = get_best_gpu()
DEVICE = torch.device(f'cuda:{best_gpu}' if torch.cuda.is_available() else 'cpu')
SAVE_MODEL = True # True if we want to sae the model
RUN_EXPERIMENTS = True  # True if we want to save logs and results
FINAL_TRAIN = True # True if we want to print and check validation measurments each epoch
models = ['Transformer','VGG','Conformer']
MODEL_NAME = 'Conformer'
assert MODEL_NAME in models
TIME = time.strftime("%Y%m%d-%H%M%S")

#Fine tuning parameters
FINE_TUNE = False


OUTPUT_LOG_DIR = os.path.join(project_dir, f'train/forcedAlignment/run_log_files/{DATASET}/{MODEL_NAME}/')
OUTPUT_OPTUNA_DIR = os.path.join(project_dir, f'train/forcedAlignment/optimization/{DATASET}/{MODEL_NAME}')
OUTPUT_DIR = os.path.join(project_dir, f'train/forcedAlignment/saved_models/{DATASET}/{MODEL_NAME}')
OUTPUT_DIR_LAST = os.path.join(project_dir, f'train/forcedAlignment/saved_models/{DATASET}/{MODEL_NAME}/last')
OUTPUT_DP_LOG_DIR = os.path.join(project_dir, f'train/forcedAlignment/dynamic_prog/results/{DATASET}/{MODEL_NAME}')


if MODEL_NAME == 'Transformer':
    TRAINING_ARGUMENTS = {
        'learning_rate': 0.000266,
        'alpha':0.77,
        'gamma':1.48,
        'lambbda':0.5,
        'num_epochs':40,
        'batch_size':32,
        'log_dir':OUTPUT_LOG_DIR
    }
elif MODEL_NAME == 'VGG':
    TRAINING_ARGUMENTS = {
        'learning_rate': 0.00022704032690633923,
        'alpha':0.71,
        'gamma':1.72,
        'lambbda':0.58,
        'num_epochs':15,
        'batch_size':512,
        'log_dir':OUTPUT_LOG_DIR
    }
elif MODEL_NAME == 'Conformer':
    TRAINING_ARGUMENTS = {
        'learning_rate': 0.000372, #0.0007329067588051209, # 0.000272, 
        'alpha':0.74,
        'gamma':1.90,
        'lambbda':0.5,
        'num_epochs':3,
        'batch_size':16,
        'log_dir':OUTPUT_LOG_DIR
    }

if DATASET == TIMIT or DATASET == BUCKEYE:
    UNSUPSEG_DIR = DATASET
else:
    UNSUPSEG_DIR = f"{DATASET}_{MODEL_TRAINED_DATASET}"
    

models_configurations = {
    'mms':
    {
    'train_folder': os.path.join(project_dir, f'train/models/MMS/MMS_train_results/MMS_train_{DATASET}_results'),
    'val_folder': os.path.join(project_dir, f'train/models/MMS/MMS_val_results/MMS_val_{DATASET}_results'),
    'test_folder': os.path.join(project_dir, f'train/models/MMS/MMS_test_results/MMS_test_{DATASET}_results')
    },
    'UnsupSeg':
    {
    'train_folder': os.path.join(project_dir, f'train/models/UnsupSeg/output_UnSupSeg_plus_cnn/{DATASET}/train'), #output_train_cnn
    'val_folder': os.path.join(project_dir, f'train/models/UnsupSeg/output_UnSupSeg_plus_cnn/{DATASET}/val'), #output_train_cnn
    'test_folder': os.path.join(project_dir, f'train/models/UnsupSeg/output_UnSupSeg_plus_cnn/{UNSUPSEG_DIR}/test') #output_train_cnn
    }
}