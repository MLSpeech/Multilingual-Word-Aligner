# MWA Detailed User Guide

## 📂 Data Preparation

Before running, organize your files so that audio names match transcript names. The tool will look for matching filenames with different extensions.
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

