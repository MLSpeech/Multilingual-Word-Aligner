from argparse import Namespace
import torch
import torchaudio
from .next_frame_classifier import NextFrameClassifier




def load_unsupseg_model(ckpt_path, device):
    """Load checkpoint and initialize model once; reuse across all files."""
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))
    model = NextFrameClassifier(hp).to(device)
    weights = {k.replace("NFC.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(weights)
    model.eval()
    return model


def get_unsupseg_embeddings(ckpt_or_model, input_wav_file, device):
    if isinstance(ckpt_or_model, str):
        model = load_unsupseg_model(ckpt_or_model, device)
    else:
        model = ckpt_or_model

    audio, sr = torchaudio.load(input_wav_file)
    assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
    audio = audio.to(device)

    with torch.no_grad():
        preds, cnn_represenataion = model(audio)
    return cnn_represenataion[0]
        #save_model_train_cnn_presentation(wav, output_dir, cnn_represenataion)

    
    