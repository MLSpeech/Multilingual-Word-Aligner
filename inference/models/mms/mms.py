import torch
import torchaudio
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any, Dict, List
from torch import Tensor
from torchaudio.functional import TokenSpan
from torchaudio.pipelines._wav2vec2.aligner import IAligner, Tokenizer, _align_emission_and_tokens
import numpy as np



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
    

def load_mms_model(device: torch.device, with_star: bool = False):
    """Load MMS_FA bundle and model once; reuse across all files."""
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star).to(device)
    model.eval()
    return bundle, model


# changed to get several files
def prepare_data(speech_file: str, words: str, with_star: bool, device: torch.device,
                 bundle=None, model=None, waveform: Tensor = None) -> tuple[Tensor, str, torch.nn.Module, Dict[str, int]]:
    if waveform is None:
        waveform, _ = torchaudio.load(speech_file)
    if bundle is None or model is None:
        bundle, model = load_mms_model(device, with_star)
    tokenizer = bundle.get_tokenizer()
    words_len = len(words)
    return waveform, words, model, tokenizer, words_len


def get_emission(waveform: Tensor, model: torch.nn.Module, device: torch.device):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    return emission.cpu()



def get_mms_embeddings(speech_file, words, device='cpu', mms_bundle=None, mms_model=None,
                       waveform: Tensor = None, **mms_config):
    try:
        repeat_mms = mms_config['mms_repeat']
        with_star = False

        waveform, transcript, model, tokenizer, words_len = prepare_data(
            speech_file, words, with_star, device,
            bundle=mms_bundle, model=mms_model, waveform=waveform)

        if words_len == 0:
            raise Exception(f"words length equal zero for mms: {transcript}")

        emission = get_emission(waveform, model, device)

        aligner = ForceAligner(blank=0, tokenizer=tokenizer)
        words_spans = aligner(emission[0], tokenizer(transcript.split()), return_as_words=True)

        confidents = [i.score for i in words_spans]
        end_times = [i.end for i in words_spans]

        end_times_divide = [int(round(end_time_i * 2)) for end_time_i in end_times]
        length_confidents = end_times_divide[-1]
        mms_confidents = np.zeros(length_confidents, dtype=np.float64)

        for item1, item2 in zip(end_times_divide, confidents):
            mms_confidents[item1 - 1] = item2
            mms_confidents[item1 - 2] = item2

        expanded_mms_confidents = np.repeat(mms_confidents[:, np.newaxis], repeat_mms, axis=1)
        return expanded_mms_confidents

    except Exception as e:
        raise Exception(f'speech file {speech_file} failed for {e}')


def extract_file_emissions_token(sentence, transcript_path, device, wav_file,
                                 mms_bundle=None, mms_model=None, waveform: Tensor = None):
    with_star = False
    waveform, transcript, model, tokenizer, words_len = prepare_data(
        wav_file, sentence, with_star, device,
        bundle=mms_bundle, model=mms_model, waveform=waveform)
    bundle = mms_bundle if mms_bundle is not None else torchaudio.pipelines.MMS_FA
    labels = bundle.get_labels()
    tokens = transcript.split()
    dictionary = {label: idx for idx, label in enumerate(labels)}
    token_indices = []
    for word in tokens:
        word_tokens = list(word)
        token_indices.append([dictionary.get(char, dictionary['*']) for char in word_tokens])

    # If there are no words in the transcript, skip
    if words_len == 0:
        print("Empty transcript, skipping.")
        return
    
    # Get emission (log-softmax probabilities) on GPU 0
    emission = get_emission(waveform, model, device)
    emission = emission.repeat_interleave(2, dim=1)[0]
    emission = torch.softmax(emission, dim=-1)
    emission  = torch.clamp(emission, 0, 1)
    emission = emission.cpu()
    return tokens, token_indices, emission

