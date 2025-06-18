import inference.configuration.constants
from inference.configuration.validate_input import UserInput, get_input_parser
from inference.configuration.get_models_configuration import get_models_configurations
from inference.models.preprocess import prepare_dataset
import os
from inference.models.predict import get_file_prediction
from inference.results_utils.graphs import extract_graphs, save_results_to_csv
from inference.models.utils import prepare_sentence, find_fit_transcript
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def align():
    
    args = get_input_parser()
    user_parameters = UserInput(**args.__dict__)
    configuration = get_models_configurations(user_parameters=user_parameters)
    
    for wav_file in tqdm(configuration['wav_input'], desc="Loading"):
        transcript_file = find_fit_transcript(wav_file, configuration['transcript_input'])
        if not len(transcript_file):
            print(f"(!) no transcript found for wav file {str(wav_file)}, continue")
            continue
        configuration['wav_file'] = str(wav_file)
        configuration['transcript_file'] = transcript_file
        sentence = prepare_sentence(configuration['transcript_file'], language=configuration['language'])
        final_embedding_batches, mask_batches = prepare_dataset(**configuration) # preprocess - as in the evaluate in our project - enter wav and transcript and extract, support .word/.wrd , json 
        
        # # add here dp return dp predictions, model prediction, etc..
        predictions, model_probs, dp_pred_frames, dp_predictions_times = get_file_prediction(final_embedding_batches, mask_batches, sentence, **configuration) # take embedding run conformer and unsup - return probabilities, trellis, graph of dp vs model
        save_results_to_csv(sentence, dp_predictions_times, **configuration)

        graphs_name = os.path.splitext(os.path.basename(configuration['wav_file']))[0]
        extract_graphs(graphs_name,predictions, model_probs, dp_pred_frames, **configuration)

def main():
    align()

if __name__ == '__main__':
    main()