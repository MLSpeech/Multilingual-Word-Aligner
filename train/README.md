# Training


### Setup environment
```bash
conda create --name MWA_train_venv --file train_venv_requirements.txt
conda activate MWA_train_venv
```

## Data Preparation
`Timit` 

Timit dataset contains train folder and test folder.

we divided the dataset for train-val-test by dividing test folder with the following:
  - test_spkrs = ["dr1", "dr2", "dr3", "dr4"]
  - val_spkrs  = ["dr5", "dr6", "dr7", "dr8"]


`Buckeye` 

dataset contains one folder for all and we divide it by speaker and divide to smaller audio.

we divided the dataset to train-val-test by dividing test folder with the following:
  - test_spkrs = ["s07", "s03", "s31", "s34"]
  - val_spkrs  = ["s40", "s39", "s36", "s25"]


When there .sph files for timit or buckeye another preprocess transformed them from sph to wav using sox 

```bash
pathes: train/forcedAlignment/utils/timit_preprocess.py and train/forcedAlignment/utils/buckeye_preprocess.py
timit: 
  /train/forcedAlignment/utils/timit_preprocess.py --spkr --source /datasets/timit/timit --target data/timit --min_phonemes 20 --/max_phonemes 50
buckeye: 
  /train/forcedAlignment/utils/buckeye_preprocess.py --spkr --source datasets/buckeye/buckeye/speech --target data/buckeye --min_phonemes 20 --max_phonemes 50

```




### Data Structure
The training script assumes that the data is structured as follows:
```bash
data
|
└───timit
|     |
|     └─ train
|     |    |   X.wav
|     |    └── X.wrd 
|     └─ val
|     |    |   Y.wav
|     |    └── Y.wrd     
|     └─ test
|     |    |   Z.wav
|     |    └── Z.wrd
└───buckeye
|     |
|     └─ train
|     |    |   X.wav
|     |    └── X.word
|     └─ val
|     |    |   Y.wav
|     |    └── Y.word     
|     └─ test
|     |    |   Z.wav
|     |    └── Z.word
```
Where X.wav is a raw waveform signal, and X.wrd/.word is its corresponding words boundaries labeled with the following format:
```bash
3050 5723 she
5723 10337 had
9190 11517 your
11517 16334 dark
16334 21199 suit
21199 22560 in
22560 28064 greasy
```


### Configuration
Prior to using our code, you need to cofigurate the training parameters and models to run under: `train/forcedAlignment/utils/constants.py`.
you need to configure `PATH_TO_DATA_DIR` the correct location of your local dataset directory.
Then define the `DATASET` you train the model on. (TIMIT or BUCKEYE) if training on onother dataset, cofigure the check points that used in UnSupSeg model.
For defining the model to run, the `MODEL_NAME` should be one of `models = ['Transformer','VGG','Conformer']`

## Running MMS and UnSupSeg for Training

Before training, **generate the MMS and UnSupSeg embeddings** by executing the following scripts: 
For UnSupSeg: 
```bash
python train/models/UnSupSeg/predict.py
```
For MMS
```bash
python train/models/MMS/mms.py
```

### Train
To run training with default hyper-parameters, run the following:
```bash
python train/forcedAlignment/train_sequence_model.py
```
To see further hyper-parameters see `train/forcedAlignment/utils/constants.py`
To train the models with diffrent sizes see `train/forcedAlignment/train_sequence_model.py`

After train is done define in `train/forcedAlignment/utils/constants.py` the `MODEL_PATHS` base on the dataset and model type. Then train the DP weights. run:
```bash
python train/forcedAlignment/dynamic_prog/train_DP.py
```
After DP finished to train, define in `train/forcedAlignment/utils/constants.py` the `DP_PATHES` base on the dataset and model type.

### Evaluation
the evaluation can be done on the model or on the DP model.
For the sequence model statistics like F1, Recall, etc. define in `train/evaluation/evaluate.py` the constants:
```python
TEST = True
VAL = False
DP = False
```

For the final model for prediction per word using the DP, define `DP=True`   . The evaluation will print alignment accuracy with diffrent tolerences.
For example:
```bash
--- Alignment Accuracy on the test [%] ---
------------------------------------------------------------------------------------------------------------------------
  t ≤ 10[msec]    |  t ≤ 15.0[msec]   |   t ≤ 20[msec]    |  t ≤ 25.0[msec]   |   t ≤ 50[msec]    |   t ≤ 100[msec]  
------------------------------------------------------------------------------------------------------------------------
      57.96       |       70.86       |       77.32       |       81.28       |       91.61       |       97.81      

```

and run:
```bash
python train/evaluation/evaluate.py
```