# MWA — Multilingual Word Aligner

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/MLSpeech)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10675-b31b1b.svg)](https://arxiv.org/abs/2606.10675)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

> ### **State-of-the-art open-source speech–text word alignment for 1000+ languages.**

---

## Citation

If you use MWA in your research, please cite:

**Multilingual Word-Level Forced Alignment with Self-Supervised Representations and Learned Dynamic Programming**  
Roy Weber, Meidan Zehavi, Rotem Rousso, Joseph Keshet  
*The 27th Annual Conference of the International Speech Communication Association (Interspeech), 2026*  
[https://arxiv.org/abs/2606.10675](https://arxiv.org/abs/2606.10675)

```bibtex
@inproceedings{weber2026multilingual,
  title     = {Multilingual Word-Level Forced Alignment with Self-Supervised Representations and Learned Dynamic Programming},
  author    = {Weber, Roy and Zehavi, Meidan and Rousso, Rotem and Keshet, Joseph},
  booktitle = {Proceedings of the 27th Annual Conference of the International Speech Communication Association (Interspeech)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.10675}
}
```

---

## What is MWA?

MWA maps spoken audio to its transcript at the **word level**, producing a
precise start and end timestamp for every word. It works for read speech,
conversational speech, and languages never seen during training.

MWA outperforms all leading speech–text aligners on the TIMIT and Buckeye
corpora. It also generalises out of the box to all 1000+ languages supported
by Meta's MMS model, including Hebrew, Dutch, German, Arabic, and many more.

![Model performance comparison](inference/images/performance.png)

---

## How it works

MWA is a three-stage ensemble pipeline:

```
Audio + Transcript
       │
       ├─── MMS-FA ──────────────────┐  character-level emission probs
       │                             │
       └─── UnsupSeg CNN ────────────┤  boundary representations
                                     │
                              Conformer (16 blocks)
                                     │  frame-wise boundary probs @ 10 ms
                                     │
                           Dynamic Programming
                                     │  penalised optimisation over
                                     │  model probs + MMS emissions
                                     │  + acoustic boundary distances
                                     │
                         Word timestamps (CSV + TextGrid)
```

1. **MMS-FA** ([Pratap et al., 2023](https://huggingface.co/facebook/mms-300m))
   — Meta's massively multilingual forced aligner provides character-level
   emission probabilities.  
2. **UnsupSeg** ([Kreuk et al., 2020](https://arxiv.org/abs/2007.13465))
   — A self-supervised CNN encoder that learns acoustic boundary cues without
   any labels.  
3. **Conformer** — A 16-block conformer trained on top of the concatenated
   features, outputting per-frame boundary probabilities at 10 ms resolution.  
4. **DP alignment** — A penalty-aware dynamic programming pass combines all
   signals to place word boundaries at the globally optimal positions.

---

## Models

| Name | HuggingFace | Trained on | Best suited for |
|---|---|---|---|
| `timit` | [MLSpeech/mwa-timit](https://huggingface.co/MLSpeech/mwa-timit) | TIMIT corpus | Read / formal speech |
| `buckeye` | [MLSpeech/mwa-buckeye](https://huggingface.co/MLSpeech/mwa-buckeye) | Buckeye corpus | Conversational / fluent speech |

Both models were trained on American English corpora but leverage MMS-FA
features, enabling alignment for all **1000+ languages** supported by MMS.

Weights are downloaded automatically from HuggingFace on first use.

---

## Supported Languages

MWA supports all **1000+ languages** covered by Meta's MMS model. For the
full list of languages and their ISO 639-3 codes, see the official MMS
language list:

**[MMS supported languages and codes](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html)**

> **Note:** MWA only requires MMS's **Language Identification (LID)** support,
> not full ASR support. This means any language listed under LID in the MMS
> coverage table is supported — a much broader set than the ASR-only languages.

MWA uses [uroman](https://github.com/isi-nlp/uroman) to romanize non-Latin
scripts automatically. Pass the ISO 639-3 code via `--language`.

**Example languages:** English (eng), Spanish (spa), French (fra), German (deu),
Arabic (ara), Hindi (hin), Mandarin Chinese (cmn), Japanese (jpn), Russian (rus),
Portuguese (por), Italian (ita), Dutch (nld), Korean (kor), Turkish (tur),
Polish (pol), Swedish (swe), Hebrew (heb), Persian (fas), Vietnamese (vie),
Swahili (swh), ...

---

## Installation

**Requirements:** Python 3.11, 16 kHz audio input.

### Conda (recommended)

```bash
git clone https://github.com/MLSpeech/Multilingual-Word-Aligner.git
cd Multilingual-Word-Aligner

conda env create -f environment.yml
conda activate Mwa_venv
pip install -e .
```

### pip / venv

```bash
git clone https://github.com/MLSpeech/Multilingual-Word-Aligner.git
cd Multilingual-Word-Aligner

python3.11 -m venv Mwa_venv
source Mwa_venv/bin/activate        # Linux / macOS
# Mwa_venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Simple CLI

```bash
mwa align <model_name> [language] --input_dir <path> --output_dir <path>
```

```bash
# Align all files in a directory (English, conversational)
mwa align buckeye eng --input_dir ./data/ --output_dir ./results/

# Language defaults to 'eng' when omitted
mwa align timit --input_dir ./data/ --output_dir ./results/
```

### Full `align_wav.py` interface

```bash
python align_wav.py \
    --wav_input        /path/to/audio/       \
    --transcript_input /path/to/transcripts/ \
    --language         eng                   \
    --model_name       buckeye               \
    --device           cuda:0                \
    --output_folder    ./results/
```

### All arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--wav_input` | path | — | `.wav`/`.flac` file **or** directory of audio files |
| `--transcript_input` | path | — | `.txt`/`.TextGrid` file **or** directory of transcript files |
| `--language` | str | `eng` | ISO 639-3 language code (see [Supported Languages](#supported-languages)) |
| `--model_name` | str | `timit` | Pretrained model: `timit` or `buckeye` |
| `--device` | str | `cpu` | PyTorch device: `cpu`, `cuda:0`, `cuda:1`, … |
| `--output_folder` | path | — | Output directory (created automatically if missing) |
| `--no_graph` | flag | off | Suppress PNG visualisation output |
| `--no_csv` | flag | off | Suppress CSV and TextGrid output |

---

## Data Preparation

Organise your files so that each audio file has a matching transcript with
the **same base name**:

```
dataset/
├── interview_01.wav
├── interview_01.txt        ← plain text, one utterance per line
├── lecture_02.flac
├── lecture_02.TextGrid     ← Praat TextGrid with a "sentence" tier
└── ...
```

**Transcript formats**

| Format | Rules |
|---|---|
| `.txt` | One sentence per file; words separated by spaces |
| `.TextGrid` | Praat format; must contain a tier named `sentence` |

**Audio requirements:** `.wav` or `.flac`, **16 kHz** sample rate.

---

## Output Reference

| File | Contents |
|---|---|
| `<name>.csv` | `Word, Start_Time, End_Time` — one row per word, times in seconds |
| `<name>.TextGrid` | Praat TextGrid with a `words` interval tier |
| `<name>_graph1.png` | Waveform (top) and frame-level boundary probabilities (bottom) with DP boundaries overlaid in blue and Conformer predictions in green |

---

## GPU Acceleration

Pass `--device cuda:0` to move all models to GPU. This is strongly recommended
for large batches. All three models (MMS, UnsupSeg, Conformer) are loaded
**once** at startup and shared across every file in the batch, so per-file
cost is pure inference with no reload overhead.

```bash
python align_wav.py \
    --wav_input        /data/corpus/ \
    --transcript_input /data/corpus/ \
    --language         eng           \
    --model_name       buckeye       \
    --device           cuda:0        \
    --output_folder    ./results/
```

---

## Running the Bundled Examples

The repository ships with two ready-to-run examples inside `inference/examples/`:

```
inference/examples/
├── english.wav          # "The car is going too fast"
├── english.txt          # transcript (.txt format)
├── english.TextGrid     # same transcript (.TextGrid format)
├── german.wav           # "wer möchte keinen Kuchen"
└── german.txt           # transcript
```

### Example 1 — English (`.txt` transcript)

```bash
python align_wav.py \
    --wav_input        inference/examples/english.wav \
    --transcript_input inference/examples/english.txt \
    --language         eng \
    --model_name       timit \
    --output_folder    results/
```

### Example 2 — English (`.TextGrid` transcript)

```bash
python align_wav.py \
    --wav_input        inference/examples/english.wav \
    --transcript_input inference/examples/english.TextGrid \
    --language         eng \
    --model_name       buckeye \
    --output_folder    results/
```

### Example 3 — German

```bash
python align_wav.py \
    --wav_input        inference/examples/german.wav \
    --transcript_input inference/examples/german.txt \
    --language         deu \
    --model_name       timit \
    --output_folder    results/
```

### Expected output

After running any example, the `results/` folder will contain:

```
results/
├── english.csv           # word-level timestamps
├── english.TextGrid      # Praat TextGrid with a "words" tier
└── english_graph1.png    # waveform + probability visualisation
```

**`english.csv`** looks like:

```
Word,Start_Time,End_Time
THE,0.0,0.12
CAR,0.12,0.31
IS,0.31,0.45
GOING,0.45,0.67
TOO,0.67,0.84
FAST,0.84,1.07
```

**`english.TextGrid`** can be opened directly in
[Praat](https://www.fon.hum.uva.nl/praat/) and contains a `words` tier with
one labelled interval per word.

---

## Acknowledgements

This work was supported by NSF DRL Grant No. 2219843 and
BSF Grant No. 2022618. We also thank Rob van Son for his
guidance and support with the IFA Corpus.
