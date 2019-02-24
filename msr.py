# -*- coding: UTF-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#@ sys.path.insert(0, '/usr/lib/python2.7/site-packages')
import os
import getopt
import scipy.io.wavfile as wavfile
import codecs

from progressbar import ProgressBar, Percentage, Bar, ETA, FormatLabel, AnimatedMarker
from msrmodel import MSRModel

"""
M-AILABS Speaker Identification tool
In order to use it, you need to first enroll speakers and create a model:

    python msr.py -c enroll -m <model-name> -e <enroll_path>

The <enroll_path> is a directory containing a subdirectory for each speaker containing audio files 
representing that speaker's speech:
    <enroll_path>
        +-----><speaker_1>
        |          +------> audio1.wav
        |          +------> audio2.wav
        |          ...
        +-----><speaker_2>
        |          +------> audio1.wav
        ...

After training, you can use the trained model to predict speakers:

    python msr.py -c predict -m <model-name> -p <predict_path> -l <predict_log_file>

The predict-path is just a directory containing .wav-files. All .wav-files will be analyzed...

Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
All Rights Reserved.

Written: 2018-01-28 10:00 CET, ISO (Imdat Solak)

"""
def ensure_mono(raw_audio):
    if raw_audio.ndim == 2:
        return raw_audio[:, 0]
    else:
        return raw_audio


def read_wav(wav_file):
    fs, signal = wavfile.read(wav_file)
    signal = ensure_mono(signal)
    return fs, signal


def enroll_speakers(model_name, enroll_path):
    speakers = {}
    model = MSRModel()
    print('Starting enrolling speakers... ')
    sys.stdout.flush()
    for root, dirs, files in os.walk(enroll_path):
        counter = 0
        num_files = len(files)
        widgets=[FormatLabel('File: %(message)s [%(value)s/'+str(num_files)+']'), ' ', Percentage(), ' ', Bar(marker='@', left='[', right=']'), ' ', ETA()]
        pBar = ProgressBar(widgets=widgets, maxval=num_files).start()
        for name in filter(lambda name: name.endswith('.wav'), files):
            counter += 1
            audio_file = os.path.join(root, name)
            speaker = os.path.basename(os.path.dirname(audio_file))
            audio_name = os.path.basename(audio_file)
            pBar.update(counter, audio_name + "@" + speaker)
            try:
                fs, signal = read_wav(audio_file)
            except:
                print('\nError at ', audio_file)
                continue
            model.enroll(speaker, fs, signal)
        pBar.finish()
    print('Starting training... ', end='')
    sys.stdout.flush()
    model.train()
    print('done')
    print('Saving model... ', end='')
    sys.stdout.flush()
    model.dump(model_name)
    print('done')


def predict_speakers(model_name, predict_path, predict_log_file):
    model = MSRModel.load(model_name)
    log_file = codecs.open(predict_log_file, 'w', 'utf-8')
    for root, dirs, files in os.walk(predict_path):
        num_files = len(files)
        widgets=[FormatLabel('File: %(message)s [%(value)s/'+str(num_files)+']'), ' ', Percentage(), ' ', Bar(marker='@', left='[', right=']'), ' ', ETA()]
        pBar = ProgressBar(widgets=widgets, maxval=num_files).start()
        counter = 0
        for name in filter(lambda name: name.endswith('.wav'), files):
            counter += 1
            audio_name = os.path.basename(name)
            pBar.update(counter, audio_name)
            audio_file = os.path.join(root, name)
            fs, signal = read_wav(audio_file)
            x = model.predict(fs, signal)
            label = x[0]
            rel_v = x[1]
            abs_v = x[2]
            rel_vals = {}
            abs_vals = {}
            for m in rel_v:
                k = m[0]
                v = m[1]
                rel_vals[k] = v
            for m in abs_v:
                k = m[0]
                v = m[1]
                abs_vals[k] = v
            label_rel_v = rel_vals[label]
            label_abs_v = abs_vals[label]
            pred_data = '{}\t{}\t{}'.format(label, label_rel_v, label_abs_v)
            others = {}
            for other_name in rel_vals.keys():
                if other_name != label:
                    others[other_name] = [rel_vals[other_name], abs_vals[other_name]]
            left_log = '{}\t{}'.format(audio_file, pred_data)
            right_log = ''
            for key in others.keys():
                a, v = others[key]
                if len(right_log)>0:
                    right_log += '\t'
                right_log += '({}\t{}\t{})'.format(key, a, v)
            print('{}\t{}'.format(left_log, right_log), file=log_file)

        pBar.finish()
    log_file.close()


def main(command, model, enroll_path, predict_path, predict_log_file):
    if command == 'enroll':
        enroll_speakers(model, enroll_path)
    elif command == 'predict' and predict_log_file is not None:
        predict_speakers(model, predict_path, predict_log_file)
    else:
        usage()


def usage():
    print('Missing command-line arguments.')
    print('Usage:')
    print('\tpython msr.py -m <model> -c <enroll|predict> <-p <predict_path> | -e <enroll_path> > -l <predict_log_file>')
    sys.exit(1)
    

if __name__ == '__main__':
    model = None
    enroll_path = None
    predict_path = None
    command = 'enroll'
    predict_log_file = 'predict.log'

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:e:p:c:l:', ['--model', '--enroll_path', '--predict_path', '--command', '--predict_log'])
    except getopt.GetoptError:
        print('Getopt error')
        usage()

    for opt, arg in opts:
        if opt in ('-m', '--model'):
            model = arg
        elif opt in ('-e', '--enroll_path'):
            enroll_path = arg
        elif opt in ('-p', '--predict_path'):
            predict_path = arg
        elif opt in ('-c', '--command'):
            command = arg
        elif opt in ('-l', '--predict_log'):
            predict_log_file = arg
    if model is not None and command is not None and (enroll_path is not None or predict_path is not None):
        main(command, model, enroll_path, predict_path, predict_log_file)
    else:
        usage()
