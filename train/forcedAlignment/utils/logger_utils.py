import logging
import os 
import time


def setup_logging(train_cfg, timestamp, model_args=None, optuna=False, run_exp=False):
    # Create timestamp for the current run and create the directories if they don't exist
    run_dir = train_cfg['log_dir']
    model = model_args['vgg_name'] if 'vgg_name' in model_args else model_args['model_type']
    sequence_size = model_args['sequence_size'] if 'sequence_size' in model_args else ''

    if optuna:
        logger = logging.getLogger(f"optuna_logger_{timestamp}")
        logger.addHandler(logging.NullHandler())
        return logger, None
    
    if not run_exp:
        # When optuna or run_exp is True, we still want to log to console but not to a file
        logger = logging.getLogger(f"exp_logger_{timestamp}")
        # Ensure we don't have any handlers attached before adding a StreamHandler for console
        if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)  # Log all messages to the console
            logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        return logger, None

    if model_args['model_type'] == 'vgg':
        # Create a unique folder for each run based on model, sequence size, and timestamp
        run_subdir = f"{model}_{sequence_size}_{timestamp}"
        full_run_dir = os.path.join(run_dir, run_subdir)
        os.makedirs(full_run_dir, exist_ok=True)
        # Set up logging
        log_file = os.path.join(full_run_dir, f"run_log_{model}_{sequence_size}_{timestamp}.log")
        logger = logging.getLogger(f"my_custom_logger_seq{model_args['sequence_size']}_vgg{model_args['vgg_name']}_{timestamp}")
        
    elif model_args['model_type'] == 'transformer':
        run_subdir = f"{model_args['model_type']}_seq{model_args['sequence_size']}_att{model_args['attention_size']}_{timestamp}"
        full_run_dir = os.path.join(run_dir, run_subdir)
        os.makedirs(full_run_dir, exist_ok=True)
        # Set up logging
        log_file = os.path.join(full_run_dir, f"run_log_{model_args['model_type']}.log")
    
        logger = logging.getLogger(f"my_custom_logger_seq{model_args['sequence_size']}_att{model_args['attention_size']}_{timestamp}")

    elif model_args['model_type'] == 'conformer':
        run_subdir = f"{model_args['model_type']}_seq{model_args['sequence_size']}_ConformerBlocks{model_args['conformer_blocks']}_heads{model_args['number_attention_heads']}_karnel{model_args['karnel_size']}_{timestamp}"
        full_run_dir = os.path.join(run_dir, run_subdir)
        os.makedirs(full_run_dir, exist_ok=True)
        # Set up logging
        log_file = os.path.join(full_run_dir, f"run_log_{model_args['model_type']}.log")
    
        logger = logging.getLogger(f"my_custom_logger_seq{model_args['sequence_size']}_ConformerBlocks{model_args['conformer_blocks']}_heads{model_args['number_attention_heads']}_karnel{model_args['karnel_size']}_{timestamp}")
    
    # Ensure handlers are not duplicated
    if logger.hasHandlers():
        logger.handlers.clear()  # Remove any existing handlers
        
    
    logger.setLevel(logging.DEBUG)  # Set the logger level
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(logging.WARNING)  # File logs warnings and above
    console_handler.setLevel(logging.DEBUG)  # Console shows all debug and above

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log the starting information
    date_str = time.strftime("%d/%m/%Y")
    time_str = time.strftime("%H:%M:%S")
    logger.warning(f"Date: {date_str}")
    logger.warning(f"Start Time: {time_str}")
    logger.warning(f"Model: {model}")
    logger.warning(f"Sequence size: {sequence_size}")
    logger.warning(f"Alpha: {train_cfg['alpha']}")
    logger.warning(f"Gamma: {train_cfg['gamma']}")
    logger.warning(f"Number of epochs: {train_cfg['num_epochs']}")
    logger.warning(f"Learning rate: {train_cfg['learning_rate']}")

    return logger, log_file

def setup_logging_DP(train_cfg, timestamp, model_args=None, run_exp=False, optuna=False):
    # Create timestamp for the current run and create the directories if they don't exist
    run_dir = train_cfg['log_dir_DP']
    model = model_args['vgg_name'] if 'vgg_name' in model_args else model_args['model_type']
    stop_condition = 'early_stop' if train_cfg['early_stop'] else 'no_early_stop'

    if optuna or ( not run_exp):
        # When optuna or run_exp is True, we still want to log to console but not to a file
        logger = logging.getLogger(f"optuna_or_exp_logger_{timestamp}")
        # Ensure we don't have any handlers attached before adding a StreamHandler for console
        if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)  # Log all messages to the console
            logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        return logger, None, None

    if model_args['model_type'] == 'vgg':
        # Create a unique folder for each run based on model, sequence size, and timestamp
        run_subdir = f"DP_{model}_{stop_condition}_{timestamp}"
        full_run_dir = os.path.join(run_dir, run_subdir)
        os.makedirs(full_run_dir, exist_ok=True)
        # Set up logging
        log_file = os.path.join(full_run_dir, f"run_log_DP_{model}_{stop_condition}_{timestamp}.log")
        logger = logging.getLogger(f"my_custom_logger_{stop_condition}_vgg{model_args['vgg_name']}_{timestamp}")
        
    elif model_args['model_type'] == 'transformer':
        run_subdir = f"DP_{model_args['model_type']}_{stop_condition}_{timestamp}"
        full_run_dir = os.path.join(run_dir, run_subdir)
        os.makedirs(full_run_dir, exist_ok=True)
        # Set up logging
        log_file = os.path.join(full_run_dir, f"run_log_DP_{model_args['model_type']}_{stop_condition}_{timestamp}.log")
        logger = logging.getLogger(f"my_custom_logger_{stop_condition}_{timestamp}")

    elif model_args['model_type'] == 'conformer':
        run_subdir = f"DP_{model_args['model_type']}_{stop_condition}_{timestamp}"
        full_run_dir = os.path.join(run_dir, run_subdir)
        os.makedirs(full_run_dir, exist_ok=True)
        # Set up logging
        log_file = os.path.join(full_run_dir, f"run_log_DP_{model_args['model_type']}_{stop_condition}_{timestamp}.log")
        logger = logging.getLogger(f"my_custom_logger_{stop_condition}_{timestamp}")
    
    # Ensure handlers are not duplicated
    if logger.hasHandlers():
        logger.handlers.clear()  # Remove any existing handlers
        
    
    logger.setLevel(logging.DEBUG)  # Set the logger level
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(logging.WARNING)  # File logs warnings and above
    console_handler.setLevel(logging.DEBUG)  # Console shows all debug and above

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log the starting information
    date_str = time.strftime("%d/%m/%Y")
    time_str = time.strftime("%H:%M:%S")
    logger.warning(f"Date: {date_str}")
    logger.warning(f"Start Time: {time_str}")
    logger.warning(f"Model: {model}")
    logger.warning(f"Early stop: {stop_condition}")
    logger.warning(f"Num Epochs: {train_cfg['num_epochs_dp']}")
    logger.warning(f"Model path: {train_cfg['model_path']}")
    logger.warning(f"C: {train_cfg['C']}")
    logger.warning(f"Max tolerence: {train_cfg['max_tolerence']}")
    logger.warning(f"Penalty gap: {train_cfg['penalty_gap']}")
    logger.warning(f"Comment: {train_cfg['comment']}")
    logger.warning(f"Comment: {train_cfg['features']}")

    return logger, log_file, full_run_dir



def setup_logging_fine_tuning(train_cfg, timestamp, model_args=None, training_arguments=None, run_exp=False, optuna=False):
    # Create timestamp for the current run and create the directories if they don't exist
    run_dir = os.path.join(train_cfg['model_path'], f'fine_tune_models')

    if optuna:
        logger = logging.getLogger(f"optuna_logger_{timestamp}")
        logger.addHandler(logging.NullHandler())
        return logger, None
    
    if not run_exp:
        # When optuna or run_exp is True, we still want to log to console but not to a file
        logger = logging.getLogger(f"optuna_or_exp_logger_{timestamp}")
        # Ensure we don't have any handlers attached before adding a StreamHandler for console
        if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)  # Log all messages to the console
            logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        return logger, None
    
    run_subdir = f'fine_tuning_{timestamp}'
    full_run_dir = os.path.join(run_dir, run_subdir)
    os.makedirs(full_run_dir, exist_ok=True)
    log_file = os.path.join(full_run_dir, f"run_log_fine_tuning_{timestamp}.log")
    logger = logging.getLogger(f"my_custom_logger_fine_tuning_{timestamp}")
    model = model_args['vgg_name'] if 'vgg_name' in model_args else model_args['model_type']
    stop_condition = 'early_stop' if train_cfg['early_stop'] else 'no_early_stop'
    
    # Ensure handlers are not duplicated
    if logger.hasHandlers():
        logger.handlers.clear()  # Remove any existing handlers
        
    
    logger.setLevel(logging.DEBUG)  # Set the logger level
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(logging.WARNING)  # File logs warnings and above
    console_handler.setLevel(logging.DEBUG)  # Console shows all debug and above

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log the starting information
    date_str = time.strftime("%d/%m/%Y")
    time_str = time.strftime("%H:%M:%S")
    logger.warning(f"Date: {date_str}")
    logger.warning(f"Start Time: {time_str}")
    logger.warning(f"Model: {model}")
    logger.warning(f"Early stop: {stop_condition}")
    logger.warning(f"Num Epochs: {training_arguments['num_epochs']}")
    logger.warning(f"Comment: {train_cfg['features']}")

    return logger, log_file


def log_details(details, logger, print_to_log=True):
    # Define column width and format for numbers (6 decimal places)
    column_width = 15
    format_string = f"{{:^{column_width}}}"

    # Prepare the header
    header = " | ".join([format_string.format(key) for key in details.keys()])

    # Prepare the values
    values = " | ".join([format_string.format(f"{value * 100:.2f}%" if key != 'OS' else f"{value:.4f}") for key, value in details.items()])
    
    if print_to_log:
        logger.warning(header)
        logger.warning(values)
    else:
        print(header)
        print(values)    



