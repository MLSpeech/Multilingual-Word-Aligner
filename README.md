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

# Basic Usage
To align audio files in a directory, use the following command structure:
```bash
mwa align <model_name (timit/buckeye)> <language> --input_dir <input_dir> --output_dir <output_dir>
```
Example:
```bash
mwa align timit eng --input_dir "/path/to/data/" --output_dir "./results"
```

Note: If language is not specified, it defaults to `eng`.

# 📖 Documentation

For detailed information on data preparation, supported languages, and advanced parameters, please refer to our:

👉 [Detailed User Guide](user_guide.md)


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


## Illustration

![Model Performance Illustration](inference/images/performance.png)