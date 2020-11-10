import os
import sys
from typing import Tuple

import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc

mean_signal_length = 17000

def get_feature_vector_from_mfcc(file_path: str, flatten: bool, mfcc_len: int = 39) -> np.ndarray:
    fs, signal = wav.read(file_path)
    sigLength = len(signal)

    if sigLength<mean_signal_length:
        pad_len = mean_signal_length - sigLength
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = sigLength - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    if flatten:
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients

def get_data(data_path: str, flatten: bool = True, mfcc_len: int = 39,
             class_labels: Tuple = ("DC", "JE", "JK", "KL")) -> \
        Tuple[np.ndarray, np.ndarray]:

    data = []
    labels = []
    names = []
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("started reading folder %s\n" % directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            filepath = os.getcwd() + '/' + filename
            feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
                                                          mfcc_len=mfcc_len,
                                                          flatten=flatten)
            data.append(feature_vector)
            if filename.startswith("a"):
                labels.append("Angry")
            elif filename.startswith("d"):
                labels.append("Disgusted")
            elif filename.startswith("f"):
                labels.append("Fearful")
            elif filename.startswith("h"):
                labels.append("Happy")
            elif filename.startswith("n"):
                labels.append("Neutral")
            else:
                if filename[0:2]=="sa":
                    labels.append("Sad")
                if filename[0:2]=="su":
                    labels.append("Surprised")
            names.append(filename)
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('..')
    os.chdir(cur_dir)
    print(np.array(labels))
    return np.array(data), np.array(labels)

get_data("./AudioData", True, 39)