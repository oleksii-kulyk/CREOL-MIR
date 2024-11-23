# Using the dsatools library from github for signal analysis
# https://github.com/MVRonkin/dsatools/blob/b0b3674beaac770021944e8844fed6c03653ef92/dsatools/_base/_imf_decomposition/_hvd.py

import dsatools
from dsatools import operators
import dsatools.utilits as ut
from dsatools import decomposition

import numpy as np

def denoize(spectrum):

  psds = decomposition.hvd(x,order=6, fpar=30)
  ut.probe(psds.sum(axis=0),title='hvd')

