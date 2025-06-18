
#buckeye preprocessing
#script by Felix Kreuk (https://github.com/felixkreuk/UnsupSeg/blob/master/scripts/preprocess_buckeye.py) (with some slight modifications)


import random
import soundfile as sf
import buckeye
from tqdm import tqdm
import numpy as np
from boltons import fileutils
import os
import os.path as osp
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--spkr', default=False, action='store_true')
parser.add_argument('--source', default=False)
parser.add_argument('--target', default=False)
parser.add_argument('--min_phonemes', type=int)
parser.add_argument('--max_phonemes', type=int)
args = parser.parse_args()



DELIMITER = ['VOCNOISE', 'NOISE', 'SIL']
FORBIDDEN = ['{B_TRANS}', '{E_TRANS}', '<EXCLUDE-name>', 'LAUGH', 'UNKNOWN', 'IVER-LAUGH', '<exclude-Name>', 'IVER']
MIN_PHONEMES = args.min_phonemes
MAX_PHONEMES = args.max_phonemes
NOISE_EDGES = 0.2
is_delim = lambda x: x.seg in DELIMITER
contain_forbidden = lambda phone_list: not set([p.seg for p in phone_list]).isdisjoint(FORBIDDEN)
path = args.source
output_path = args.target
train_path = osp.join(output_path, "train")
val_path = osp.join(output_path, "val")
test_path = osp.join(output_path, "test")


files = []
segments = []
file_counter = 0

os.makedirs(output_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

def timit_create_new_file(wav_file):
	file_name = Path(wav_file).name
	file_name = file_name.replace(Path(file_name).suffix, "")
	conv_name = Path(wav_file).parent.name
	chunk_name = Path(wav_file).parent.parent.name
	new_file = f"{chunk_name}_{conv_name}_{file_name}.wav"
	return new_file

if args.spkr:

	test_spkrs = ["dr1", "dr2", "dr3", "dr4"]
	val_spkrs  = ["dr5", "dr6", "dr7", "dr8"]

	all_wavs = list(fileutils.iter_find_files(path, "*.wav", include_dirs=True))
	# wavs_train = list(fileutils.iter_find_files(train_dir, "*.wav", include_dirs=True))
	for i,wav_file in enumerate(all_wavs):
		chunk_name = Path(wav_file).parent.parent.name
		new_file = timit_create_new_file(wav_file)
		
		ouput_folder = 'train'
		if 'train' in wav_file:
			new_folder = os.path.join(output_path, 'train')	
		elif 'test' in wav_file and chunk_name in val_spkrs:
			new_folder = os.path.join(output_path, 'val')	
		elif 'test' in wav_file and chunk_name in test_spkrs:
			new_folder = os.path.join(output_path, 'test')
		else:
			continue
		
		if i % 10 == 9:
			print(f"train files: {len(os.listdir(os.path.join(output_path, 'train')))}, \
		 val files: {len(os.listdir(os.path.join(output_path, 'val')))},  test files: {len(os.listdir(os.path.join(output_path, 'test')))}")
			
		

		input_wav_file = wav_file #fine
		input_phn_file = wav_file.replace("wav", "phn") #fine
		input_word_file = wav_file.replace("wav", "wrd") #need to be .word

		os.system(f"cp {input_wav_file} {new_folder}/{new_file}")
		os.system(f"cp {input_phn_file} {new_folder}/{new_file.replace('wav','phn')}")
		os.system(f"cp {input_word_file} {new_folder}/{new_file.replace('wav','wrd')}")



else:
	
	splits = [0.8, 0.9, 1.0]
	all_wavs = list(fileutils.iter_find_files(path, "*.wav", include_dirs=True))
	random.shuffle(all_wavs)
	for i, wav in enumerate(all_wavs):
		
		new_file = timit_create_new_file(wav)
		
		
		if i < len(all_wavs) * splits[0]:
			new_folder = os.path.join(output_path, 'train')
		elif len(all_wavs) * splits[0] <= i and i < len(all_wavs) * splits[1]:
			new_folder = os.path.join(output_path, 'val')
		else:
			new_folder = os.path.join(output_path, 'test')

		os.system(f"mv {wav} {new_folder}/{new_file}")
		os.system(f"mv {wav.replace('wav', 'phn')} {new_folder}/{new_file.replace('wav','phn')}")
		os.system(f"mv {wav.replace('wav', 'wrd')} {new_folder}/{new_file.replace('wav','wrd')}")
