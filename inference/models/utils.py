import os
import json
import torch
from inference.models.conformer.conformer import initialize_conformer, Conformer
from praatio import textgrid
import uroman as ur

def prepare_sentence(transcript_file, language='eng'):
    
    available_languages = ['ara', 'bel', 'bul', 'deu', 'ell', 'eng', 'fas', 'grc', 'ell', 'eng', 'heb', 'kaz', 'kir', 'lav', 'lit', 'mkd', 'mkd2', 'oss', 'pnt', 'pus', 'rus', 'srp', 'srp2', 'tur', 'uig']
    uroman = ur.Uroman()
    if '.TextGrid' in transcript_file: 
        tg = textgrid.openTextgrid(transcript_file, includeEmptyIntervals=False)
        word_tier = tg._tierDict["sentence"]
        words = [entry.label for entry in word_tier.entries][0]
        
        # print('word_tier.__dict__')
        # [(entry.label, entry.start, entry.end) for entry in word_tier.entries]
    elif '.txt' in transcript_file:
        with open(transcript_file, 'r') as file:
            lines = file.readlines()
            words = " ".join([line.strip() for line in lines])
            
    else:
        raise Exception(".TextGrid and .txt supported only, for format example look under examples folder") 
    
    if language in available_languages:
        words = uroman.romanize_string(words, lcode=language)
        words = words.lower()
        return words
    else:
        raise Exception(f"MMS needs to get engilish character supported languages can be found here\n https://github.com/isi-nlp/uroman/tree/master?tab=readme-ov-file \
                        \n of the following languages {available_languages}")

def find_fit_transcript(wav_file, transcriptions):
    transcript_basename = wav_file.name[:-4]
    transcriptions_filtered = [i for i in transcriptions if i.name == transcript_basename+".txt" or i.name == transcript_basename+".TextGrid" ]
    if not len(transcriptions_filtered):
        return ""
    transcript_file = str(transcriptions_filtered[0])
    return transcript_file
    

def initialize_model(model='Conformer', model_args={}):
    if 'conformer' in model.lower() :
        model = initialize_conformer(model_args)
    else:
        model = None 
    return model

def load_model(model_dir, **model_attributes):
    # Define paths for the weights and configuration files
    device = model_attributes['device']
    if model_attributes['model_name'].lower() == 'timit':
        model = Conformer.from_pretrained("MLSpeech/mwa-timit")
    elif model_attributes['model_name'].lower() == 'buckeye':
        model = Conformer.from_pretrained("MLSpeech/mwa-buckeye")
    else:
        raise Exception('(error) model_name is not supported enter model from ["timit", "buckeye"]')
    
    model = model.to(device)
    model.eval()

    return model

