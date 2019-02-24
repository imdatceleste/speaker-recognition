# -*- encoding: UTF-8 -*-
from __future__ import print_function
import time
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import defaultdict
from scipy.io import wavfile
import numpy as np
import cPickle as pickle
import traceback as tb

from features import mix_feature
from VAD import VAD

try:
    from gmmset import GMMSetPyGMM as GMMSet
    from gmmset import GMM
except:
    print( "Warning: failed to import fast-gmm, use gmm from scikit-learn instead", file=sys.stderr)
    from skgmm import GMMSet, GMM

CHECK_ACTIVE_INTERVAL = 1       # seconds

class MSRModel(object):

    UBM_MODEL_FILE = None

    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()
        self.vad = VAD()

    def init_noise(self, fs, signal):
        """
        init vad from environment noise
        """
        self.vad.init_noise(fs, signal)

    def filter(self, fs, signal):
        """
        use VAD (voice activity detection) to filter out silence part of a signal
        """
        ret, intervals = self.vad.filter(fs, signal)
        orig_len = len(signal)

        if len(ret) > orig_len / 3:
            # signal is filtered by VAD
            return ret
        return np.array([])

    def enroll(self, name, fs, signal):
        """
        add the signal to this person's training dataset
        name: person's name
        """
        feat = mix_feature((fs, signal))
        self.features[name].extend(feat)

    def _get_gmm_set(self):
        if self.UBM_MODEL_FILE and os.path.isfile(self.UBM_MODEL_FILE):
            try:
                from gmmset import GMMSetPyGMM
                if GMMSet is GMMSetPyGMM:
                    return GMMSet(ubm=GMM.load(self.UBM_MODEL_FILE))
            except Exception as e:
                print("Warning: failed to import gmmset. You may forget to compile gmm:")
                print(e)
                print("Try running `make -C src/gmm` to compile gmm module.")
                print("But gmm from sklearn will work as well! Using it now!")
            return GMMSet()
        return GMMSet()

    def train(self):
        self.gmmset = self._get_gmm_set()
        start = time.time()
        for name, feats in self.features.iteritems():
            self.gmmset.fit_new(feats, name)
        # print(time.time() - start, " seconds")

    def predict(self, fs, signal):
        """
        return a label (name)
        """
        try:
            feat = mix_feature((fs, signal))
        except Exception as e:
            print(tb.format_exc())
            return None
        return self.gmmset.predict_one(feat)

    def dump(self, fname):
        """ dump all models to file"""
        self.gmmset.before_pickle()
        with open(fname, 'w') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'r') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R

