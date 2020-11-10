import scipy.io.wavfile as wav
import contextlib
import os
import sys
from typing import Tuple

def getLength(filepath):
    fs, signal = wav.read(filepath)
    return len(signal)


class_labels: Tuple = ("DC", "JE", "JK", "KL")

totalDuration=0
count=0
cur_dir = os.getcwd()
sys.stderr.write('curdir: %s\n' % cur_dir)
os.chdir("AudioData")
for i, directory in enumerate(class_labels):
    sys.stderr.write("started reading folder %s\n" % directory)
    os.chdir(directory)
    for filename in os.listdir('.'):
        filepath = os.getcwd() + '/' + filename
        print (filepath)
        count+=1
        totalDuration += getLength(filepath)
    os.chdir('..')
os.chdir(cur_dir)
print(totalDuration/count)
