# Mwa

We present the MWA - Multi Word Aligner , a new open source model for speech-text alignment. We developed an ensemble-based word - alignment algorithm composed of several state-of-the-art speech representation models.<br> The selected representation from these models is fed into a neural sequence model - Conformer, which then outputs frame-wise probabilities at a 10ms resolution.<br> Finally, dynamic programming is used to perform the final alignment, refining the boundaries and ensuring accurate word segmentation. <br> We compared the proposed model to the leading speech-text aligners model today using TIMIT and Buckeye corpora. Results suggest that out model surpasses all the leading models and reaches state-of-the-art performance on both data sets.<br> Furthermore, we evaluated the resulting model on languages that were not seen during the training phase (Hebrew and Dutch).<br> Our models can be found in huggingface in the following models pages: MLSpeech/mwa-timit, MLSpeech/mwa-buckeye<br>

# Quick start:

```bash
git clone https://github.com/MLSpeech/MWA-multilingual-word-aligner.git
```

Python3.11 environment for running (conda/pip/uv):

### conda
```
conda create --name Mwa_venv --file requirements.txt python=3.11
conda activate Mwa_venv
pip install -r requirements.txt
```

### python venv
```
python3.11 -m venv Mwa_venv
Linux - source Mwa_venv/bin/activate
Windows - Mwa_venv\Scripts\activate
pip install -r requirements.txt
```

### Run:
```bash
Input example:
python align_wav.py --wav_input "your_folder/Mwa/examples/" --transcript_input "your_folder/Mwa/examples/" --language "eng" --model_name "timit" --device "cuda:2" --output_folder "results"

Or:
python align_wav.py --wav_input "your_folder/Mwa/examples/english.wav" --transcript_input "your_folder/Mwa/examples/english.txt" --language "eng" --model_name "buckeye" --output_folder "results"
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

<!-- ![align_wav.py Parameters](inference/images/help_image.png) -->
(!) Use align_wav.py --help for Further explanation

| Argument                  | Type   | Description                                                                       
| ---------------------     | ------ | ---------------------------------------------------------------------------------------------------------------------------------------  
| `--wav_input`             | `str`  | 📂 Path to the folder containing `.wav` audio files or file with .wav posix.
| `--transcript_input`      | `str`  | 📂 Path to the folder containing transcription files (e.g., `.txt`, `.csv`). (or text file)
| `--output_folder`         | `str`  | 📁 Directory where results will be saved. Folder will be created if it doesn't exist. In order to disable result extraction use --no_graph or --no_csv additional flags
| `--language`              | `str`  | 🌍 (Optional) Language code for processing. options {ara,bel,bul,deu,ell,eng,fas,grc,ell,eng,heb,kaz,kir,lav,lit,mkd,mkd2,oss,pnt,pus,rus,srp,srp2,tur,uig} Default: `eng`. For more information use --help flag.
| `--model_name`            | `str`  | 🤖 timit/buckeye models are supported - timit model was trained on genre of spoken language and buckeye on fluent speech (conversation)
| `--device`                | `str`  | 🖥️ in case GPU resources are available you can use device name ("cuda:0")  to improve performances
| `--no_graph, --no_csv`    | `str`  | 🐞 flags to disable file extraction in results



## Illustration

![Model Performance Illustration](inference/images/performance.png)