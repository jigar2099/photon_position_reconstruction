import numpy as np
import pandas as pd
import pickle
from src.mc.MonteCarlo import MonteCarlo
from src.mc.MC import MC_set
from src.mc.shape_ext import shape_generator
from src.utils import tools as fg

# Required files: MC.py, monte_carlo.py and pulse shape file in 
# .npy format(here plateauless_uniq_pulses.npy)

def generator(samp_size,max_rate,name, train_exe):
    
    '''
    This function generates MC data for training, testing and validation,
    and save them to "./DATASETS/sub_folder_name/" path.
    
    All dataset are generated using given pulse shapes, unlike separate splitting
    of set of pulse shapes for train, test and val data generation.
    
    Test and val data has similar size(however can be changed in this function)
    The ratio between training dataset size and validation dataset size is 0.33 here.
    
    All datasets has specific format: range of rate(in photon number) with desired
    number of sample examples for each rate.
    '''
    
    name1 = name.split('.')[0]
    folder_name = 'sh06x2'
    val_exe = test_exe = int(train_exe*0.33)
    
    
    x = MC_set(name, max_rate, train_exe, samp_size)
    remove_duplicates = x.remove_duplicate_pulse()
    train,evaluation = x.train_test_val_split(60,40)
    train_x, train_y, train_x1, train_y1, CORR_pattern, CH0_ind, CH1_ind = x.get_mc_set(remove_duplicates)
    np.save(f'./DATASETS/{folder_name}/lab_ch0_shapes_train_x_m_{train_exe}.npy',train_x)
    np.save(f'./DATASETS/{folder_name}/lab_ch0_shapes_train_y_m_{train_exe}.npy',train_y)
    
    np.save(f'./DATASETS/{folder_name}/lab_ch1_shapes_train_x_m_{train_exe}.npy',train_x1)
    np.save(f'./DATASETS/{folder_name}/lab_ch1_shapes_train_y_m_{train_exe}.npy',train_y1)
    
    with open(f'./DATASETS/{folder_name}/lab_ch0_1_shapes_train_CORR_pattern_{train_exe}', 'wb') as fp:
        pickle.dump(CORR_pattern, fp)
    with open(f'./DATASETS/{folder_name}/lab_shapes_train_CH0_ind_{train_exe}', 'wb') as ch0p:
        pickle.dump(CH0_ind, ch0p)
    with open(f'./DATASETS/{folder_name}/lab_shapes_train_CH1_ind_{train_exe}', 'wb') as ch1p:
        pickle.dump(CH1_ind, ch1p) 
        
    
    x = MC_set(name, max_rate, test_exe, samp_size)
    remove_duplicates = x.remove_duplicate_pulse()
    train,evaluation = x.train_test_val_split(60,40)
    test_x, test_y, test_x1, test_y1, CORR_pattern, CH0_ind, CH1_ind = x.get_mc_set(remove_duplicates)
    np.save(f'./DATASETS/{folder_name}/lab_ch0_shapes_test_x_m_{train_exe}.npy',test_x)
    np.save(f'./DATASETS/{folder_name}/lab_ch0_shapes_test_y_m_{train_exe}.npy',test_y)
    
    np.save(f'./DATASETS/{folder_name}/lab_ch1_shapes_test_x_m_{train_exe}.npy',test_x1)
    np.save(f'./DATASETS/{folder_name}/lab_ch1_shapes_test_y_m_{train_exe}.npy',test_y1)
    
    with open(f'./DATASETS/{folder_name}/lab_ch0_1_shapes_test_CORR_pattern_{train_exe}', 'wb') as fp:
        pickle.dump(CORR_pattern, fp)
    with open(f'./DATASETS/{folder_name}/lab_shapes_test_CH0_ind_{train_exe}', 'wb') as ch0p:
        pickle.dump(CH0_ind, ch0p)
    with open(f'./DATASETS/{folder_name}/lab_shapes_test_CH1_ind_{train_exe}', 'wb') as ch1p:
        pickle.dump(CH1_ind, ch1p)
    
    
    x = MC_set(name, max_rate, val_exe, samp_size)
    remove_duplicates = x.remove_duplicate_pulse()
    train,evaluation = x.train_test_val_split(60,40)
    val_x, val_y, val_x1, val_y1, CORR_pattern, CH0_ind, CH1_ind = x.get_mc_set(remove_duplicates)
    np.save(f'./DATASETS/{folder_name}/lab_ch0_shapes_val_x_m_{train_exe}.npy',val_x)
    np.save(f'./DATASETS/{folder_name}/lab_ch0_shapes_val_y_m_{train_exe}.npy',val_y)
    
    np.save(f'./DATASETS/{folder_name}/lab_ch1_shapes_val_x_m_{train_exe}.npy',val_x1)
    np.save(f'./DATASETS/{folder_name}/lab_ch1_shapes_val_y_m_{train_exe}.npy',val_y1)
    
    with open(f'./DATASETS/{folder_name}/lab_ch0_1_shapes_val_CORR_pattern_{train_exe}', 'wb') as fp:
        pickle.dump(CORR_pattern, fp)
    with open(f'./DATASETS/{folder_name}/lab_shapes_val_CH0_ind_{train_exe}', 'wb') as ch0p:
        pickle.dump(CH0_ind, ch0p)
    with open(f'./DATASETS/{folder_name}/lab_shapes_val_CH1_ind_{train_exe}', 'wb') as ch1p:
        pickle.dump(CH1_ind, ch1p)
        
        
    
    print(train_x.shape,val_x.shape,test_x.shape)
    print(train_y.shape,val_y.shape,test_y.shape)
