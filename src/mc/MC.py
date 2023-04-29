# import lib
import numpy as np
import sys
sys.path.append("..")
from src.mc.MonteCarlo import MonteCarlo as mc
from tqdm import tqdm

class MC_set:
    def __init__(self, filepath, max_Rate, exe, sample_size):
        self.filepath = np.load(filepath)
        self.max_Rate= max_Rate
        self.exe = exe
        self.sample_size = sample_size
    def remove_duplicate_pulse(self):
        if len(self.filepath.shape)<2:
            x_all = self.filepath.reshape(int(self.filepath.shape[0]/100),100)
        else:
            x_all = self.filepath.copy()
        # Remove duplicates
        shapes = [x_all[i][x_all[i]!=0] for i in range(x_all.shape[0])]# creates list of shapes
        uniq_shapes = [list(j) for j in set(map(tuple,shapes))]# sublist match in the form of tuple
        print(f"Total unique shapes:{len(uniq_shapes)}")
        return uniq_shapes
    def train_test_val_split(self, train, test):
        y = self.remove_duplicate_pulse()
        train_lim = int(len(y)*(train/100))
        test_lim = int(len(y)*(test/100))
        #val_lim = int(len(y)*(val/100))
        return y[:train_lim], y[train_lim:]#, y[train_lim+test_lim:]

    def get_mc_set(self, shape_list):
        data_X = []; data_X1 = []
        data_y = []; data_y1 = []
        CORR_pattern = []
        CH1_ind = []
        CH2_ind = []
        for i in tqdm(range(self.max_Rate)):
            #print('samle_len : ', len(shape_list))
            #print('rate : ', i)
            #print('samp_size : ', self.sample_size)
            x, y, x1, y1, cor_pat, ch1_ind, ch2_ind = mc(shape_list, i, self.exe, self.sample_size).MC2()
            data_X.append(x); data_X1.append(x1)
            data_y.append(y); data_y1.append(y1)
            CORR_pattern.append(cor_pat)
            CH1_ind.append(ch1_ind)
            CH2_ind.append(ch2_ind)
        #data_X, data_y = mc(shape_list,2,10,100).MC2()
        data_X = np.array(data_X); data_X = data_X.reshape(self.exe*self.max_Rate, self.sample_size)
        data_y = np.array(data_y); data_y = data_y.reshape(self.exe*self.max_Rate, self.sample_size)

        data_X1 = np.array(data_X1); data_X1 = data_X1.reshape(self.exe*self.max_Rate, self.sample_size)
        data_y1 = np.array(data_y1); data_y1 = data_y1.reshape(self.exe*self.max_Rate, self.sample_size)
        return data_X, data_y, data_X1, data_y1, CORR_pattern, CH1_ind, CH2_ind
