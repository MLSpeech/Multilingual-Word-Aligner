from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Any
import os
import torchaudio
from typing import Optional, Any
import argparse
from inference.models.utils import prepare_sentence
from pathlib import Path
import torch

def get_input_parser():
    parser = argparse.ArgumentParser(description="Parser for MWA aligner, Input example can be found in .Readme")
    parser.add_argument("--wav_input", type=str, default="Mwa/inference/examples/", \
                        help="Wav input can be either file or folder with .wav files")
    parser.add_argument("--transcript_input", type=str, default="Mwa/inference/examples/", \
                        help="Transcript input can be either file of .txt, .TextGrid or folder with .txt , .TextGrid files with same names as wav files")
    parser.add_argument("--language", type=str, choices=['ara', 'bel', 'bul', 'deu', 'ell', 'eng', 'fas', 'grc', 'ell', 'eng', 'heb', 'kaz', 'kir', 'lav', 'lit', 'mkd', 'mkd2', 'oss', 'pnt', 'pus', 'rus', 'srp', 'srp2', 'tur', 'uig'],\
                        default="eng", help="""language of the transcript, MMS anticipate english character, we use https://github.com/isi-nlp/uroman to transcribe other languages to MMS format \n
                        if your language is not supported in this list pay attention to convert transcription before to MMS text format, more explanation can be found here: https://huggingface.co/docs/transformers/en/model_doc/mms  \n
                        (Arabic, Belarusian, Bulgarian, English, German, Ancient Greek, Modern Greek, Pontic Greek, Hebrew, Kazakh, Kyrgyz, Latvian, Lithuanian, Macedonian, Ossetian, Persian, Russian, Serbian, Turkish, Ukrainian, Uyghur or Yiddish)
                        """, required = False)
    parser.add_argument("--model_name", type=str, choices=["timit", "buckeye"], default="buckeye", help="Type of model to use")
    parser.add_argument("--device", type=str, default="cuda:2", help="Running resource device name for torch.device(device)") #cpu
    parser.add_argument("--output_folder", type=str, default='results', help="Output folder for graphs and .csv results")
    parser.add_argument("--no_graph", action='store_true', help="extract no graph of models")
    parser.add_argument("--no_csv", action='store_true', help="extract no csv for word timestamps")

    args = parser.parse_args()
    return args

class UserInput(BaseModel):
    wav_input: str
    transcript_input: str
    language: str
    model_name: str
    device: str
    output_folder: str
    no_graph: bool
    no_csv: bool
    
    
    @field_validator("output_folder", mode="after")
    def validate_output_folder(cls, output_folder):
        if not os.path.exists(output_folder):
            print(f"output folder {output_folder} does not exist, creating folder.")
            os.makedirs(output_folder, exist_ok=True)
        return output_folder
    
    @field_validator("wav_input", mode="after")
    def validate_wav_input(cls, wav_input):
        path = Path(wav_input)
        if not path.exists() or (path.is_file() and ".wav" not in wav_input):
            raise ValueError(f"Wav input does not exist: {wav_input}")
        if path.is_file():
            return [path]
        elif path.is_dir():
            return sorted(p for p in path.iterdir() if p.suffix == ".wav")
        else:
            raise ValueError(f"Invalid input format: {wav_input} check input again")

    @field_validator("transcript_input", mode="after")
    def validate_transcript_file(cls, transcript_input):
        path = Path(transcript_input)
        if not path.exists() or (path.is_file() and ".txt" not in transcript_input and ".TextGrid" not in transcript_input):
            raise ValueError(f"Transcript input does not exist: {transcript_input}")
        if path.is_file():
            return [path]
        elif path.is_dir():
            return sorted(p for p in path.iterdir() if p.suffix == ".txt" or p.suffix == ".TextGrid")
        else:
            raise ValueError(f"Invalid input format: {transcript_input} check input again")

    @field_validator("device", mode="after")
    def validate_device(cls, device):            
        try:
            device = torch.device(device)
            return device
        except Exception as e:
            raise ValueError(f"Invalid device, enter device as cpu, cude:1, etc. torch.device(device) has failed")
    


def main():
    
    UserInput(wav_file="dr1_mkls0_sx447.wav",
              transcript_file="dr1_mkls0_sx447.wrd",
              language="en", model_name="timit", device="cpu", output_folder=os.getcwd())

if __name__ == '__main__':
    main()