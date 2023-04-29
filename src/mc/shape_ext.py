import pathlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class shape_generator:
    def __init__(self, calib_file, high_rate_file, path_to_save=pathlib.Path().resolve()):
        self.calib_file = calib_file
        self.high_rate_file = high_rate_file
        self.path_to_save = path_to_save
        
    def generator(self, threshold=0, bin_num=2147483600):
        arr0 = -np.load(self.calib_file)[:bin_num] + threshold
        arr = np.where(arr0<0,0,arr0)
        arr_sub = arr-1
        arr_w = np.where(arr_sub<0,0,arr_sub)#/np.max(arr_sub)
        arr_threshold = arr_w.reshape(np.int(arr_w.shape[0]/100),100)

        rate = []
        ind = []
        val = []
        for i in range(arr_threshold.shape[0]):
            peak_val, peak_ind = find_peaks(arr_threshold[i], height=0)
            rate.append(len(peak_val))
            val.append(peak_val)
            ind.append(peak_ind)

        rate = np.asarray(rate)
        ind = np.asarray(ind)
        val = np.asarray(val)
        df = pd.DataFrame(np.hstack((rate,ind)))

        first_non_zero_ind = np.argwhere(arr_threshold[:,0]!=0)
        mod_arr1 = np.delete(arr_threshold,first_non_zero_ind,0)
        last_non_zero_ind = np.argwhere(mod_arr1[:,99]!=0)
        mod_arr2 = np.delete(mod_arr1,last_non_zero_ind,0)

        rate_side = []
        ind_side = []
        val_side = []

        for i in range(mod_arr2.shape[0]):
            peak_ind, peak_val = find_peaks(mod_arr2[i], height=0)
            rate_side.append(len(peak_ind))
            ind_side.append(peak_ind)
            val_side.append(peak_val)

        rate_side = np.asarray(rate_side)
        ind_side = np.asarray(ind_side)
        val_side = np.asarray(val_side)
        #df1 = pd.DataFrame(np.hstack((rate_side,ind_side)))

        rate_side_ind = np.argwhere(rate_side==1) # indices where there is only 1 photon in rate_side samples
        x = mod_arr2[rate_side_ind,:] # using row index create matrix whose samples have only 1 photon
        x=x.reshape(x.shape[0],100) # reshape the matrix
        return x
    
    def scalar(self,x, threshold=0, bin_num=2147483600):
        high_rate = -np.load(self.high_rate_file)[:bin_num] + threshold#[:2147483600]
        x_scale = x/1#np.max(high_rate)
        name = str(self.path_to_save)+self.high_rate_file[-24:].split('.')[0]+'_shapes.npy'
        np.save(name,x_scale)
        return x, name