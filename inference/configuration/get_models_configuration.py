from argparse import Namespace
import yaml
from inference.configuration.constants import CONFIG_PATH, INFERENCE_PART_DIR
import os 


def get_models_configurations(config_path = CONFIG_PATH, user_parameters={}):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    user_parameters = user_parameters.__dict__
    # output_folder = user_parameters['output_folder']
    # user_parameters['output_folder'] = os.path.join(INFERENCE_PART_DIR,output_folder)
    
    model_type = user_parameters['model_name'].lower()
    config['model_mapping'] = config['models_mapping'][model_type]
    
    final_configuration =  {**config['model_mapping'], **user_parameters}
    
    return final_configuration    


