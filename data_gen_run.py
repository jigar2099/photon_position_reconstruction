import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("..")
from src.mc.MonteCarlo import MonteCarlo
from src.mc.MC import MC_set
from src.mc.shape_ext import shape_generator
from src.utils import tools as fg
from src.mc import data_gen

# Required files: MC.py, monte_carlo.py and pulse shape file in 
# .npy format(here plateauless_uniq_pulses.npy)


train_exe = 50
name = './src/data/plateauless_uniq_pulses_345.npy'
samp_size = 256
max_num_photon = 60
data_gen.generator(samp_size, max_num_photon, name, train_exe)