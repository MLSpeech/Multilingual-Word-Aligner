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

import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt
import torchaudio.functional as F
import os
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any, Dict, List
from torch import Tensor
from torchaudio.functional import TokenSpan
from torchaudio.pipelines._wav2vec2.aligner import IAligner, Tokenizer, _align_emission_and_tokens
from train.forcedAlignment.utils.constants import DATASET, PATH_TO_DATA_DIR, DATASETS_MAPPING

print(torch.__version__)
print(torchaudio.__version__)
device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(device)

def get_device():
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")



def _flatten(list_: List[List[Any]]) -> List[Any]:
    return list(chain.from_iterable(list_))


def _unflatten(list_: List[Any], lengths: List[int]) -> List[List[Any]]:
    assert len(list_) == sum(lengths), "Lengths must sum to list length"
    it = iter(list_)
    return [list(islice(it, l)) for l in lengths]



@dataclass
class WordSpan:
    """WordSpan()
    Word with time stamps and score.
    """

    word: str
    """The token"""
    start: int
    """The start time (inclusive) in emission time axis."""
    end: int
    """The end time (exclusive) in emission time axis."""
    score: float
    """The score of the this word."""

    def __len__(self) -> int:
        """Returns the time span"""
        return self.end - self.start

    def __repr__(self):
        return f"{self.word:<20}[{self.start:7d}, {self.end:7d}){self.score:>6.2f}"
    

def merge_tokens(tokens: Tensor, scores: Tensor, blank: int = 0) -> List[TokenSpan]:
    """Removes repeated tokens and blank tokens from the given CTC token sequence.

    Args:
        tokens (Tensor): Alignment tokens (unbatched) returned from :py:func:`forced_align`.
            Shape: `(time, )`.
        scores (Tensor): Alignment scores (unbatched) returned from :py:func:`forced_align`.
            Shape: `(time, )`. When computing the token-size score, the given score is averaged
            across the corresponding time span.
            

    Returns:
        list of TokenSpan

    Example:
        >>> aligned_tokens, scores = forced_align(emission, targets, input_lengths, target_lengths)
        >>> token_spans = merge_tokens(aligned_tokens[0], scores[0])
    """
    if tokens.ndim != 1 or scores.ndim != 1:
        raise ValueError("`tokens` and `scores` must be 1D Tensor.")
    if len(tokens) != len(scores):
        raise ValueError("`tokens` and `scores` must be the same length.")

    # Compute the difference between consecutive tokens.
    diff = torch.diff(
        tokens, prepend=torch.tensor([-1], device=tokens.device), append=torch.tensor([-1], device=tokens.device)
    )
    # Compute the change points and mask out the points where the new value is blank
    changes_wo_blank = torch.nonzero((diff != 0)).squeeze().tolist()

    tokens = tokens.tolist()
    spans = [
        TokenSpan(token=token, start=start, end=end, score=scores[start:end].mean().item())
        for start, end in zip(changes_wo_blank[:-1], changes_wo_blank[1:])
        if (token := tokens[start]) != blank
    ]
    return spans


def merge_word_tokens(tokens_spans: List[List[TokenSpan]], i2c: Dict[int, str]) -> List[List[WordSpan]]:
    """Merges the tokens spans into a list of WordSpan.

    Args:
        tokens_spans (List[TokenSpan]): Tokens spans returned by :py:func:`merge_tokens`.
        i2c (Dict[int, str]): Dictionary of token to character.

    Returns:
        List[WordSpan]: list of WordSpan
    """

    words_span = []
    for word_spans in tokens_spans:
        word = "".join(i2c[t.token] for t in word_spans)
        start = word_spans[0].start
        end = word_spans[-1].end
        score = sum(s.score * len(s) for s in word_spans) / sum(len(s) for s in word_spans)
        words_span.append(WordSpan(word=word, start=start, end=end, score=score))
    return words_span



class ForceAligner(IAligner):
    def __init__(self, tokenizer: Tokenizer, blank: int = 0):
        self.blank = blank
        self.tokenizer = tokenizer
        self.decoder = {v: k for k, v in tokenizer.dictionary.items()}

    def __call__(
        self, emission: Tensor, tokens: List[List[int]], return_as_words: bool = False
    ) -> List[List[TokenSpan]] | List[List[WordSpan]]:
        """Aligns the given emission and tokens.

        Args:
            emission (Tensor): Emission tensor. 2D tensor of shape: `(time, n_classes)`.
            tokens (List[List[int]]): List of tokens. Shape: `(n_words, n_tokens)`.
            return_as_words (bool, optional): If True, returns the aligned tokens as words.

        Raises:
            ValueError: If the input emission is not 2D.

        Returns:
            List[List[TokenSpan]] | [List[WordSpan]]: List of token spans or word spans.

        Example:
            >>> text = "i had that curiosity beside me at this moment"
            >>> transcript = text.split()
            >>> words_spans = aligner(emission, tokenizer(transcript), return_as_words=True)
        """
        if emission.ndim != 2:
            raise ValueError(f"The input emission must be 2D. Found: {emission.shape}")

        aligned_tokens, scores = _align_emission_and_tokens(emission, _flatten(tokens), self.blank)
        spans = merge_tokens(aligned_tokens, scores)
        spans = _unflatten(spans, [len(ts) for ts in tokens])
        if return_as_words:
            spans = merge_word_tokens(spans, self.decoder)
        return spans

    def decode(self, tokens: int | List[int]) -> str:
        """Decode the given tokens into a string.

        Args:
            tokens (int | List[int]): Tokens to be decoded, either a single token or a list of tokens.

        Returns:
            str: Decoded string
        """
        if isinstance(tokens, int):
            return self.decoder[tokens]
        return "".join(self.decoder[t] for t in tokens)
    

# changed to get several files
def prepare_data(speech_file: str, transcript_file: str, with_star: bool, device: torch.device) -> tuple[Tensor, str, torch.nn.Module, Dict[str, int]]:
    torch.cuda.empty_cache()
    waveform, _ = torchaudio.load(speech_file)
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star)  # Load model on CPU first
    model = model.to(device)  # Then move to GPU
    tokenizer = bundle.get_tokenizer()
    
    # Parse the transcript file and extract only the words (ignoring numbers)
    with open(transcript_file, 'r') as file:
        lines = file.readlines()

    words = [line.split()[-1] for line in lines if line.strip()]  # Ignore empty lines
    words_len = len(words)
    
    # Concatenate the extracted words into a transcript
    transcript = ' '.join(words)    
    return waveform, transcript, model, tokenizer, words_len


def get_emission(waveform: Tensor, model: torch.nn.Module, device: torch.device):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    return emission


def plot_emission(emission):
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()



def get_files_list(input_dir):
    """
    preparing the datasets files
    """
    # Initialize lists to store the file names
    speech_files = []
    transcript_files = []
    posix_mms = DATASETS_MAPPING[DATASET]['word_posix']
    # Iterate through the files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            # If it's a .wav file, add it to the speech files list
            speech_files.append(filename)
        elif filename.endswith(posix_mms):
            # If it's a .txt file, add it to the transcript files list
            transcript_files.append(filename)

    # Sort the lists to ensure they match up correctly
    speech_files.sort()
    transcript_files.sort()
    return speech_files, transcript_files


def main():
    device = get_device()
    print(device)
    

    for mode in ['train','val','test']:
        i = 0
        input_dir = os.path.join(PATH_TO_DATA_DIR, f'/{DATASET}/{mode}/')
        output_dir = os.path.join(project_dir, f'train/models/MMS/MMS_{mode}_results/MMS_{mode}_{DATASET}_results')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        speech_files, transcript_files = get_files_list(input_dir)

        for speech_file, transcript_file in zip(speech_files, transcript_files):
            try:
                i+=1
                # Extract the name of the WAV file without the extension
                wav_file_name = os.path.splitext(os.path.basename(speech_file))[0]
                # Define the output file name
                output_file_name = os.path.join(output_dir,f"{wav_file_name}_mms.txt")
                if os.path.exists(output_file_name):
                    print(f"{output_file_name} already exists")
                    continue
                with_star = False #True
                full_path_speech_file = os.path.join(input_dir,f"{speech_file}")
                full_path_transcript_file = os.path.join(input_dir,f"{transcript_file}")
                if i%100 == 0:
                    print(f"{i} of {len(speech_files)},{full_path_speech_file}")
                
                waveform, transcript, model, tokenizer, words_len = prepare_data(full_path_speech_file, full_path_transcript_file, with_star, device) 
                
                if words_len == 0:
                    print("Error: Number of words is zero!")
                    continue
                else:
                    
                    
                    emission = get_emission(waveform, model, device)
                    
                    #plot_emission(emission[0])
                    
                    bundle = torchaudio.pipelines.MMS_FA
                    tokenizer = bundle.get_tokenizer()
                    
                    
                    aligner = ForceAligner(blank=0, tokenizer=tokenizer)
                    words_spans = aligner(emission[0], tokenizer(transcript.split()), return_as_words=True)
                    #print(f"Transcript: {transcript}")
                    #print(f"Words spans:")
                    #for word_span in words_spans:
                    #    print(word_span)
                    #print("=" * 60)
                
                    # Open the output file for writing
                    with open(output_file_name, 'w') as output_file:
                        output_file.write(f"Transcript: {transcript}\n")
                        output_file.write("Words spans:\n")
                        for word_span in words_spans:
                            output_file.write(f"{word_span}\n")
                        output_file.write("=" * 60 + "\n")
                        
                    print(f"Results saved to {output_file_name}")
                
                    #for text in [TRANSCRIPT, TRANSCRIPT_EOS_STAR, TRANSCRIPT_MIDDLE_STAR, TRANSCRIPT_BOS]:
            except Exception as e:
                print(f'speech file {speech_file} failed for {e}')




if __name__ == "__main__":
    main()