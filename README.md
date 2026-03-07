# Mwa

We present the MWA - Multi Word Aligner , a new open source model for speech-text alignment. 

We developed an ensemble-based word - alignment algorithm composed of several state-of-the-art speech representation models.

The selected representation from these models is fed into a neural sequence model - Conformer, which then outputs frame-wise probabilities at a 10ms resolution.

Finally, dynamic programming is used to perform the final alignment, refining the boundaries and ensuring accurate word segmentation. 

We compared the proposed model to the leading speech-text aligners model today using TIMIT and Buckeye corpora. Results suggest that out model surpasses all the leading models and reaches state-of-the-art performance on both data sets.

Furthermore, we evaluated the resulting model on languages that were not seen during the training phase (Hebrew, Dutch and German).

Our models can be found in huggingface in the following models pages: 
  - [Timit model](https://huggingface.co/MLSpeech/mwa-buckeye)
  - [Buckeye model](https://huggingface.co/MLSpeech/mwa-timit)


# Quick start:

```bash
git clone https://github.com/MLSpeech/MWA-multilingual-word-aligner.git
```

Python3.11 environment for running ([conda](https://docs.conda.io/en/latest/)/pip/uv):

### conda
```
conda env create -f environment.yml
conda activate Mwa_venv
```

### python venv
```
python3.11 -m venv Mwa_venv
Linux - source Mwa_venv/bin/activate
Windows - Mwa_venv\Scripts\activate
pip install -r requirements.txt
```


### User guide and examples:

- Before running create audio-text files scheme with the following format with audio names matching transcript names
```
dataset/
├── audio1.wav
├── audio1.txt
├── audio2.flac
├── audio2.TextGrid
├── audio3.wav
├── audio2.txt
...
```
- Text should be with one line containing the text seperated by " " or TextGrid with text tag (Example in inference/examples/english.TextGrid file)
- Then check match audio language you want to align [supported langauges](https://huggingface.co/facebook/mms-1b-all#supported-languages)
- Add resources with "--device"
- Choose one of the models: 
  - [timit](https://huggingface.co/MLSpeech/mwa-buckeye)
  - [buckeye](https://huggingface.co/MLSpeech/mwa-timit)
- Run Mwa to align your audio 

- Code examples:

```bash
Input example:
python align_wav.py --wav_input "<wav_folder>" --transcript_input "<transcript_folder>" --language "eng" --model_name "timit" --device "cuda:2" --output_folder "results"

Or:
python align_wav.py --wav_input "your_folder/Mwa/examples/english.wav" --transcript_input "your_folder/Mwa/examples/english.txt" --language "eng" --model_name "buckeye" --output_folder "results"

Or:
python align_wav.py --wav_input "your_folder/Mwa/examples/english.wav" --transcript_input "your_folder/Mwa/examples/english.TextGrid" --language "eng" --model_name "timit" --device "cuda:2"

Or:
python align_wav.py --wav_input "your_folder/Mwa/examples/german.wav" --transcript_input "your_folder/Mwa/examples/german.txt" --language "deu" --model_name "timit" --output_folder "results"
```

<!-- Licenses:
```bash
This is from Felix we need to 
@article{kreuk2020self,
  title={Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation},
  author={Kreuk, Felix and Keshet, Joseph and Adi, Yossi},
  journal={arXiv preprint arXiv:2007.13465},
  year={2020}
}
``` -->





## MWA Usage

Use align_wav.py --help for Further explanation

| Argument                  | Type   | Description                                                                       
| ---------------------     | ------ | ---------------------------------------------------------------------------------------------------------------------------------------  
| `--wav_input`             | `str`  | 📂 Path to the folder containing `.wav/.flac` audio files or file with `.wav/.flac` posix.
| `--transcript_input`      | `str`  | 📂 Path to the folder containing transcription files (e.g., `.txt/.TextGrid`, `.csv`). (or text file)
| `--output_folder`         | `str`  | 📁 Directory where results will be saved. Folder will be created if it doesn't exist. In order to disable result extraction use --no_graph or --no_csv additional flags
| `--language`              | `str`  | 🌍 Language code for processing. options {ara,bel,bul,deu,ell,eng,fas,grc,ell,eng,heb,kaz,kir,lav,lit,mkd,mkd2,oss,pnt,pus,rus,srp,srp2,tur,uig} Default: `eng`. For more information use --help flag.
| `--model_name`            | `str`  | 🤖 timit/buckeye models are supported - timit model was trained on genre of spoken language and buckeye on fluent speech
| `--device`                | `str`  | 🖥️ in case GPU resources are available you can use device name ("cuda:0")  to improve performances
| `--no_graph, --no_csv`    | `str`  | 🐞 flags to disable file extraction in results




## Illustration

![Model Performance Illustration](inference/images/performance.png)