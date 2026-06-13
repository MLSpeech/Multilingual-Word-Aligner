import inference.configuration.constants
from inference.configuration.validate_input import UserInput, get_input_parser
from inference.configuration.get_models_configuration import get_models_configurations
from inference.models.preprocess import prepare_dataset
import os
import time
from inference.models.predict import get_file_prediction
from inference.results_utils.graphs import extract_graphs, save_results_to_csv, save_results_to_textgrid
from inference.models.utils import prepare_sentence, find_fit_transcript, load_model
from inference.models.mms.mms import load_mms_model
from inference.models.unsupSeg.unsupseg_classifier import load_unsupseg_model
from inference.models.dp_algorithm.extract_features import Features_DP
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def align():

    args = get_input_parser()
    user_parameters = UserInput(**args.__dict__)
    configuration = get_models_configurations(user_parameters=user_parameters)

    model = load_model(**configuration)
    model_name = configuration.get('model_name') or configuration.get('model_path') or 'model'
    print(f"Loaded model: {model_name}")

    print("Loading MMS model...")
    mms_bundle, mms_model = load_mms_model(configuration['device'])
    configuration['mms_bundle'] = mms_bundle
    configuration['mms_model'] = mms_model

    print("Loading UnsupSeg model...")
    unsupseg_ckpt = os.path.join(inference.configuration.constants.INFERENCE_PART_DIR,
                                 configuration['unsupseg_ckpt'])
    configuration['unsupseg_model'] = load_unsupseg_model(unsupseg_ckpt, configuration['device'])

    # Pre-build DP objects once; reused across all files
    configuration['_dp_features_obj'] = Features_DP(configuration['dp_features'])
    configuration['_w_floats'] = list(configuration['w'])

    file_times = []
    total_start = time.time()

    for wav_file in tqdm(configuration['wav_input'], desc="Processing", unit="file"):
        transcript_file = find_fit_transcript(wav_file, configuration['transcript_input'])
        if not len(transcript_file):
            tqdm.write(f"(!) no transcript found for wav file {str(wav_file)}, continue")
            continue

        file_start = time.time()

        configuration['wav_file'] = str(wav_file)
        configuration['transcript_file'] = transcript_file
        sentence = prepare_sentence(configuration['transcript_file'], language=configuration['language'])
        final_embedding_batches, mask_batches = prepare_dataset(**configuration)

        predictions, model_probs, dp_pred_frames, dp_predictions_times = get_file_prediction(model, final_embedding_batches, mask_batches, sentence, **configuration)
        save_results_to_csv(sentence, dp_predictions_times, **configuration)

        graphs_name = os.path.splitext(os.path.basename(configuration['wav_file']))[0]
        extract_graphs(graphs_name, predictions, model_probs, dp_pred_frames, **configuration)

        save_results_to_textgrid(sentence, dp_predictions_times, **configuration)

        file_times.append(time.time() - file_start)

    total_elapsed = time.time() - total_start
    mean_time = np.mean(file_times) if file_times else 0.0
    print(f"\nDone.")
    print(f"Total time  : {total_elapsed:.1f}s")
    print(f"Mean / file : {mean_time:.1f}s  ({len(file_times)} file(s))")


def main():
    align()

if __name__ == '__main__':
    main()