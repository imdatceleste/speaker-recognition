# M-AILABS Speaker Recognition

This is a tool to separate speakers in a list of audio files.

It is quite easy to use. You have a two-step-approach:
1. Enroll Speakers
2. Recognize Speakers

## Usage
You have two usage models as well: enroll and predict

### Enroll

    python msr.py -m mymodel.model -c enroll -e data

### Predict

    python msr.py -m mymodel.model -c predict -p test -l mylogfile.csv

## Enrolling Speakers
If you want to enroll speakers, you need to create a directory containing the training data (wav-files). If we call the directory ``data`` and we want to identify two different speakers, the structure is like this:

     data
        +--> speaker_1
        |       +---> file1.wav
        |       +---> file2.wav
        |       ...
        +--> speaker_2
        |       +---> file1.wav
        |       ...

Then you can start the tool to enroll them using:

    python msr.py -m <model_name> -c enroll -e data

That's all

## Predicting
For prediction, equally, you should save all your wav-files you want to analyze into a directory (e.g. 'test') and then you can start prediction.

    python msr.py -m <model_name> -c predict -p test [-l mylogfile.csv]

## Requirements
You must have python 2.7 and various libraries installed (numpy, scipy). Additionally, you need to install ``pyssp``.

Lastly, please install scikits.talkbox

Optional: fast-gmm and bob (see below)

### Installation scikits.talkbox
Install the scikits.talkbox from the ``contrib``-directory using the installation instructions there...

## Speed-up

--WORK IN PROGRESS--

You can speed things a little if you compile the library ``gmm`` in ``msrmodel/gmm``. Normally, the only thing you have to do is go in there and type:

    make

Everything else is already setup. But it may be buggy :-)

--END WORK IN PROGRESS--

You can speed-up things even more if you install ``bob``-packages:

    pip install bob.extension bob.blitz bob.core bob.sp bob.ap



That's all folks, hope you have fun...

---
Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH - All Rights Reserved.
# speaker-recognition
