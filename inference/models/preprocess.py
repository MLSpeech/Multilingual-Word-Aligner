import inference.configuration.constants as constants
import numpy as np
import torchaudio
from inference.models.mms.mms import get_mms_embeddings
from inference.models.unsupSeg.unsupseg_classifier import get_unsupseg_embeddings
from inference.models.utils import prepare_sentence
import os

def contrast_normalization(arr):
    # Get the min and max values from the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Avoid division by zero if all values are the same
    if arr_min == arr_max:
        return np.zeros_like(arr)  # Return an array of zeros if no contrast can be applied

    # Normalize the array to the range [0, 1]
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    
    return normalized_arr

def pad_and_batch_transformer_and_conformer(final_embedding, sequence_size=200):
    """
    Pad the final embeddings and labels to be divisible by batch size and split into batches.
    
    Args:
        final_embedding (numpy.ndarray): The final embedding matrix (frames x embedding_dim).
        labels (numpy.ndarray): The labels corresponding to the embeddings.
        sequence_size (int): The batch size to pad and split to. Default is 200.
        
    Returns:
        numpy.ndarray: A numpy array of batches for embeddings.
        numpy.ndarray: A numpy array of batches for labels.
        numpy.ndarray: A numpy array of batches for masks.
    """
    frames_length = final_embedding.shape[0]
    embedding_dim = final_embedding.shape[1]  # size of each embedding vector

    final_embedding_batches = []
    mask_batches = []

    if frames_length < sequence_size:
        # Calculate the amount of padding needed
        padding_length = sequence_size - frames_length

    
        # Create a padding array of shape (padding_length, embedding_dim) filled with -inf
        padding = np.full((padding_length, embedding_dim), 0.0) #-np.inf
        final_embedding_padded = np.vstack([final_embedding, padding])
        final_embedding_batches.append(final_embedding_padded)

        mask = np.ones(sequence_size, dtype=bool)
        mask[frames_length:] = False
        mask_batches.append(mask)
  
    else:
        for start in range(0, frames_length - sequence_size + 2, sequence_size): 
            sub_embedding = final_embedding[start:start + sequence_size]
            final_embedding_batches.append(sub_embedding)

            # Create the mask: 
            mask = np.ones(sequence_size, dtype=bool)
            mask_batches.append(mask)

        if (frames_length + 1 ) % sequence_size != 0:
            start = frames_length - sequence_size
            sub_embedding = final_embedding[start:]
            final_embedding_batches.append(sub_embedding)

            mask = np.ones(sequence_size, dtype=bool)
            num_zeros = sequence_size - (frames_length % sequence_size)
            mask[:num_zeros] = False
            mask_batches.append(mask)

    # Convert batches to numpy arrays before returning
    final_embedding_batches = np.array(final_embedding_batches)
    mask_batches = np.array(mask_batches)

    return final_embedding_batches, mask_batches



def prepare_dataset(**model_arguments):
    speech_file = model_arguments['wav_file']
    transcript_file = model_arguments['transcript_file']
    words = prepare_sentence(transcript_file, language=model_arguments['language'])

    # Load waveform once; share with extract_file_emissions_token later
    waveform, _ = torchaudio.load(speech_file)
    model_arguments['_waveform'] = waveform

    mms_confidents = get_mms_embeddings(
        speech_file, words,
        device=model_arguments['device'],
        mms_bundle=model_arguments.get('mms_bundle'),
        mms_model=model_arguments.get('mms_model'),
        waveform=waveform,
        mms_repeat=model_arguments['mms_repeat'],
    )
    unsupseg_input = model_arguments.get('unsupseg_model') or \
        os.path.join(constants.INFERENCE_PART_DIR, model_arguments['unsupseg_ckpt'])
    UnsupSeg_cnn_represenataion = get_unsupseg_embeddings(unsupseg_input, speech_file, model_arguments['device'])
    if not isinstance(UnsupSeg_cnn_represenataion, np.ndarray):
        UnsupSeg_cnn_represenataion = UnsupSeg_cnn_represenataion.cpu().detach().numpy()

    min_embedd_shape = min(mms_confidents.shape[0], UnsupSeg_cnn_represenataion.shape[0])
    mms_confidents = mms_confidents[:min_embedd_shape]
    UnsupSeg_cnn_represenataion = UnsupSeg_cnn_represenataion[:min_embedd_shape]

    norm_UnsupSeg_cnn_represenataion = contrast_normalization(UnsupSeg_cnn_represenataion)
    final_embedding = np.hstack((norm_UnsupSeg_cnn_represenataion, mms_confidents))
    final_embedding_batches, mask_batches = pad_and_batch_transformer_and_conformer(
        final_embedding, sequence_size=model_arguments['sequence_size'])
    return final_embedding_batches, mask_batches
    