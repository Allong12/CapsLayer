import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')

from config import cfg
from capslayer.utils import load_data

%run vectorCapsNet.py

%cd ..

model = CapsNet(height=28, width=28, channels=1, num_label=10)

print(repr(model.digitCaps))