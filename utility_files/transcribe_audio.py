# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:41:20 2021

@author: Samarth Gupta (samarthgupta24@gmail.com)
"""

import nltk
import librosa
import torch
import soundfile as sf
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
nltk.download('punkt')

import deepspeech
import wave
import numpy as np
import pydub
from pydub import AudioSegment
from tqdm import tqdm

class Transcribe:
    def __init__(self,model,device='cpu',tokenizer=None,filename=None):
        self.model = model
        self.tokenizer = tokenizer
        self.filename = filename
        self.device = device
    
    def asr_transcript(self):
        """
        Returns the transcript of the input audio recording

        Output: Transcribed text
        Input: Huggingface tokenizer, model and wav file
        """
        audio_chunks = self.slit_audio(self.filename)
        w2v_transcript = []
        self.model = self.model.to(self.device)
        for i, chunk in enumerate(tqdm(audio_chunks)):
            #chunk = silence_chunk + chunk
            chunk.export('temp.wav', format="wav")
            #read the file
            speech, samplerate = sf.read('temp.wav')
            #make it 1-D
            if len(speech.shape) > 1: 
                  speech = speech[:,0] + speech[:,1]
            #Resample to 16khz
            if samplerate != 16000:
                  speech = librosa.resample(speech, samplerate, 16000)
            #tokenize
            input_values = self.tokenizer(speech, return_tensors="pt").input_values.to(self.device)
            #input_values = input_values.cuda()
            #take logits
            logits = self.model(input_values).logits
            #take argmax (find most probable word id)
            predicted_ids = torch.argmax(logits, dim=-1)
            #get the words from the predicted word ids
            transcription = self.tokenizer.decode(predicted_ids[0])
            transcription = self.correct_uppercase_sentence(transcription.lower())
            w2v_transcript.append(transcription)
        text = '. '.join(w2v_transcript)
        return text
    
    def ds_transcript(self):
        audio_chunks = self.slit_audio(self.filename)
        ds_trans1 = []
        #silence_chunk = AudioSegment.silent(duration=500)
        for i, chunk in enumerate(tqdm(audio_chunks)):
            #chunk = silence_chunk + chunk
            chunk.export('temp.wav', format="wav")
            w = wave.open('temp.wav', 'r')
            rate = w.getframerate()
            frames = w.getnframes()
            buffer = w.readframes(frames)
            data16 = np.frombuffer(buffer, dtype=np.int16)
            text = self.model.stt(data16)
            text = self.correct_uppercase_sentence(text.lower())
            ds_trans1.append(text)
        text = ' '.join(ds_trans1)
        return text
    
    def match_target_amplitude(self,sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
    
    def slit_audio(self,filename):
        from pydub import AudioSegment
        from pydub.silence import split_on_silence

        sound_file = AudioSegment.from_wav(filename)
        sound_file = self.match_target_amplitude(sound_file, -20.0)
        sound_file = sound_file.set_frame_rate(16000)
        sound_file = sound_file.set_channels(1)
        #sound_file = sound_file.
        audio_chunks = split_on_silence(sound_file, 
            # must be silent for at least half a second
            min_silence_len = 1000,
        #     # consider it silent if quieter than -16 dBFS
            silence_thresh=-35
        )
        return audio_chunks
    
    def correct_uppercase_sentence(self,input_text): 
        """
        Returns the corrected sentence
        """  
        sentences = nltk.sent_tokenize(input_text)
        return (' '.join([s.replace(s[0],s[0].capitalize(),1) for s in sentences]))

