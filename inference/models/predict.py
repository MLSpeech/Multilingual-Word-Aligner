import torch
import numpy as np
from .utils import load_model
from inference.models.dp_algorithm.utils import create_tensor_from_indices_masked, find_optimal_positions_with_penalty
from inference.models.utils import prepare_sentence
from inference.models.mms.mms import extract_file_emissions_token
from tqdm import tqdm


def create_tensor_from_indices(y, masks, conf_model):
    if 'vgg' in conf_model['model_name']:
        tensor = torch.zeros((len(masks), 1, 1), dtype=torch.float32)
        for index in y:
            tensor[index][0][0] = 1
    else:
        tensor = torch.zeros((len(masks), conf_model['sequence_size'], 1), dtype=torch.float32)
        # Count the number of False values
        false_count = np.count_nonzero(masks[-1] == False)
        # Set the corresponding indices in the tensor to 1
        for index in y:
            # Calculate the row and column based on the index in the list
            row = index // conf_model['sequence_size']  # Row is index divided by 300 (integer division)
            col = index % conf_model['sequence_size']   # Column is the remainder of the division
            if (row == (len(masks) - 1)) and (len(masks) != 1):
                col = col + false_count
            
            # Set the value at the specific index to 1
            tensor[row][col][0] = 1

    return tensor

def get_model_prediction(model, embeddings, masks, **configuration):

    device = configuration['device']
    model_name = configuration["model_name"]

    with torch.inference_mode():
        embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32).to(device)
        masks_tensor = torch.as_tensor(masks).to(device)
        outputs = model(input_ids=embedding_tensor) #shape: torch.Size([5, 80, 1])
        if 'vgg' in model_name.lower():
            middle_index = embedding_tensor.shape[1] // 2
            embedding_tensor = embedding_tensor[:, middle_index:middle_index+1, :]  # Shape becomes [X, 1, Y]
        # Apply the mask to filter the tensors by indexing
        masked_embeddings = embedding_tensor[masks_tensor.bool()].unsqueeze(0) #shape:([356, 192])
        masked_outputs = outputs[masks_tensor.bool()].unsqueeze(0) #shape:([356, 1])
        probabilities = torch.sigmoid(masked_outputs).view(-1).cpu() #shape:(356,)
        predictions = (probabilities >= 0.5).float()
        
    return masked_embeddings, probabilities, predictions



def get_file_prediction(model, embeddings, masks, sentence,**configuration):

    masked_embeddings, probabilities, predictions = get_model_prediction(model, embeddings, masks, **configuration)

    _, token_indices, emissions = extract_file_emissions_token(
        sentence=sentence, transcript_path=configuration['transcript_file'],
        device=configuration['device'], wav_file=configuration['wav_file'],
        mms_bundle=configuration.get('mms_bundle'),
        mms_model=configuration.get('mms_model'),
        waveform=configuration.get('_waveform'),
    )

    dp_predictions_times, _, _ = find_optimal_positions_with_penalty(len(masked_embeddings[0]), sentence.count(' ')+1, probabilities,
                            sentence, masked_embeddings[0].cpu(), token_indices, emissions, **configuration)
    
    dp_pred_frames = create_tensor_from_indices_masked(dp_predictions_times, predictions_vector_size=len(predictions))

    return predictions, probabilities, dp_pred_frames, dp_predictions_times
    