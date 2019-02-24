# -*- coding: UTF-8 -*-
from __future__ import print_function
# File: __init__.py
# Date: Sat Nov 29 21:42:15 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys

try:
    import BOB as MFCC
except Exception as e:
    print(e, file=sys.stderr)
    print( "Warning: failed to import Bob, will use a slower version of MFCC instead.", file=sys.stderr)
    import MFCC

# import MFCC
import LPC
import numpy as np

def get_extractor(extract_func, **kwargs):
    def f(tup):
        return extract_func(*tup, **kwargs)
    return f

def mix_feature(tup):
    mfcc = MFCC.extract(tup)
    lpc = LPC.extract(tup)
    if len(mfcc) == 0:
        print("ERROR.. failed to extract mfcc feature:", len(tup[1]), file=sys.stderr)
    return np.concatenate((mfcc, lpc), axis=1)
