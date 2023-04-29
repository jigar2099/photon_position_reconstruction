import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle
import seaborn as sns
import random
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
import tqdm
import matplotlib.ticker as mtick


def pulse_height(uniq_shapes):
    H = [np.max(np.array(uniq_shapes[i])) for i in range(len(uniq_shapes))]
    sns.distplot(H, bins=30, kde=True, color='black')
    plt.title('Pulse Height')
    plt.savefig('pulse_height_platauless_shapes.png')
    plt.show()

def pulse_width(uniq_shapes):
    H = [len(np.array(uniq_shapes[i])[np.nonzero(np.array(uniq_shapes[i]))]) for i in range(len(uniq_shapes))]
    sns.distplot(H, bins=30, kde=True, color='black')
    plt.savefig('pulse_width_platauless_shapes.png')
    plt.title('Pulse Width')
    plt.show()

def pulse_skewness(uniq_shapes):
    H = [skew(np.array(uniq_shapes[i])) for i in range(len(uniq_shapes))]
    ax = sns.distplot(H, bins=30, kde=True, color='black')
    kde_x, kde_y = ax.lines[0].get_data()
    #plotting the two lines
    p1 = plt.axvline(x=-0.5,color='#EF9A9A', alpha=0.01)
    p2 = plt.axvline(x=0.5,color='#EF9A9A', alpha=0.01)
    p3 = plt.axvline(x=1,color='#EF9A9A', alpha=0.01)


    #ax.fill_between(kde_x, kde_y, where=(kde_x<0) | (kde_x>1), 
    #                interpolate=True, color='red', alpha= 0.5)
    #plt.yscale('log')
    plt.title('Pulse Skewness')
    plt.show()
    
def pulse_kurtosis(uniq_shapes):
    H = [kurtosis(np.array(uniq_shapes[i])) for i in range(len(uniq_shapes))]
    ax = sns.distplot(H, bins=30, kde=True, color='black')
    kde_x, kde_y = ax.lines[0].get_data()
    #plotting the two lines
    p1 = plt.axvline(x=-0.5,color='#EF9A9A', alpha=0.01)
    p2 = plt.axvline(x=0.5,color='#EF9A9A', alpha=0.01)
    p3 = plt.axvline(x=1,color='#EF9A9A', alpha=0.01)


    #ax.fill_between(kde_x, kde_y, where=(kde_x<0) | (kde_x>1), 
    #                interpolate=True, color='red', alpha= 0.5)
    #plt.yscale('log')
    plt.title('Pulse Kurtosis')
    plt.show()

def pulse_data(uniq_shape):
    '''
    This function creates a csv file wich contains different parameters
    of the pulses from calibration data
    This includes plateau size, pulse height and width, skweness and kurtosis
    It takes shapes extracted from calibration measurements in the format of
    numpy array for example shape=(100,100)
    '''
    puls_idx = []; plat_s = []; puls_h = []; puls_W = []; skw = []; krt = []
    for i in range(len(uniq_shape)):
        #print(uniq_shape[i])
        peaks, peak_plateaus = find_peaks(uniq_shape[i], plateau_size=0)
        plat_size = peak_plateaus['plateau_sizes'][0]
        pulse_height = np.max(np.array(uniq_shape[i]))
        pulse_width = len(np.array(uniq_shape[i])[np.nonzero(np.array(uniq_shape[i]))])
        skewness = skew(uniq_shape[i])
        kurt = kurtosis(uniq_shape[i])
        puls_idx.append(i)
        plat_s.append(plat_size)
        puls_h.append(pulse_height)
        puls_W.append(pulse_width)
        skw.append(skewness)
        krt.append(kurt)
    df = pd.DataFrame({'pulse_idx': puls_idx,'plateau_size': plat_s,'pulse_height': puls_h,'pulse_width': puls_W,'skewness': skw,'kurtosis': krt})
    return df




def plat_plot(df, bins=50, img_save_path=None):
    fig, axs = plt.subplots(3, 2,figsize=(13,7))
    axs[0, 0].hist(df.loc[df['plateau_size']==1]['pulse_height'], bins=bins)
    axs[0, 0].set_title('Plateau size = 1')
    axs[0, 1].hist(df.loc[df['plateau_size']==2]['pulse_height'], bins=bins)
    axs[0, 1].set_title('Plateau size = 2')
    axs[1, 0].hist(df.loc[df['plateau_size']==3]['pulse_height'], bins=bins)
    axs[1, 0].set_title('Plateau size = 3')
    axs[1, 1].hist(df.loc[df['plateau_size']==4]['pulse_height'], bins=bins)
    axs[1, 1].set_title('Plateau size = 4')
    axs[2, 0].hist(df.loc[df['plateau_size']==4]['pulse_height'], bins=bins)
    axs[2, 0].set_title('Plateau size = 4')
    axs[2, 1].hist(df.loc[df['plateau_size']==5]['pulse_height'], bins=bins)
    axs[2, 1].set_title('Plateau size = 5')

    axs[0,0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    axs[0,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    axs[1,0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    axs[1,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    for ax in axs.flat:

        ax.set(xlabel='Pulse height', ylabel='count')

    plt.tight_layout()
    plt.savefig(img_save_path+'shaula_calib_ch0_EDA_plateau.png')
    
def skew_kurt(df, plat_size=1, img_save_path=None):
    df_1 = df.loc[df['plateau_size']==plat_size]
    sns.histplot(
        df_1, x="skewness", y="kurtosis",
        bins=30, discrete=(False, False), log_scale=(False, True),
        cbar=True, cbar_kws=dict(shrink=.97),
    )
    plt.savefig(f"{img_save_path}calib_data_skew_vs_kurtosis_plat_size_{plat_size}.png")




def shape_filt(df,uniq_shape,platau_removal=True):
    '''
    This function removes shapes with platau if platau_removal is True(Bydefault)
    if set to False then it simply removes duplicates from uniq_shape
    It takes shapes extracted from calibration measurements in the format of numpy
    array, and created dataframe from "pulse_data" function
    '''
    if platau_removal is True:
        two = df.loc[df['plateau_size']==2]['pulse_idx'].to_list()
        three = df.loc[df['plateau_size']==3]['pulse_idx'].to_list()
        four = df.loc[df['plateau_size']==4]['pulse_idx'].to_list()
        five = df.loc[df['plateau_size']==5]['pulse_idx'].to_list()
        removal_idx = two+three+four+five
        platauless_shape = np.delete(uniq_shape, [removal_idx], axis=0)
        
        shapes = [platauless_shape[i][platauless_shape[i]!=0] for i in range(platauless_shape.shape[0])]# creates list of shapes
        uniq_shapes = [list(j) for j in set(map(tuple,shapes))]# sublist match in the form of tuple
        return platauless_shape, uniq_shapes
    else:
        shapes = [uniq_shape[i][uniq_shape[i]!=0] for i in range(uniq_shape.shape[0])]# creates list of shapes
        uniq_shapes = [list(j) for j in set(map(tuple,shapes))]# sublist match in the form of tuple
        return uniq_shape
    

def filter_shapes_based_on_plat_size(df, uniq_shape, platau_removal=True, lst=[2,3,4,5], save_path=None):
    '''
    This function removes shapes with platau if platau_removal is True(Bydefault)
    if set to False then it simply removes duplicates from uniq_shape
    It takes shapes extracted from calibration measurements in the format of numpy
    array, and created dataframe from "pulse_data" function
    '''
    df1 = df.copy()
    if platau_removal is True:
        ind_lst = [df1.loc[df1['plateau_size']==i]['pulse_idx'].to_list() for i in lst]
        concat_ind_lst = [j for i in ind_lst for j in i]
        platauless_shape = np.delete(uniq_shape, concat_ind_lst, axis=0)
        save_name1 = ''.join([str(k) for k in lst])
        save_name0 = 'plateauless_uniq_pulses_'
        np.save(save_path + save_name0 + save_name1 + '.npy', platauless_shape)
        return platauless_shape, uniq_shape
    else:
        #shapes = [uniq_shape[i][uniq_shape[i]!=0] for i in range(uniq_shape.shape[0])]# creates list of shapes
        #uniq_shapes = [list(j) for j in set(map(tuple,shapes))]# sublist match in the form of tuple
        return uniq_shape
    
    
    
   
    
def get_consecutive_numbers(num, n, m):
    '''
    get n numbers before and after a provided number with a difference of m 
    '''
    return list(range(num-n*m, num+(n+1)*m, m))

def replace_with_unique_element(arr, n=1):
    """
    Randomly selects a unique element from a list or NumPy array `arr`,
    selects `n` indices in `arr` (excluding the index of the selected unique
    element), and replaces the values at these selected indices with the selected
    unique element.
    """
    # Select unique element
    unique_element = np.random.choice(np.unique(arr))

    # Select indices to replace
    indices_to_replace = np.random.choice(np.delete(np.arange(len(arr)), np.where(arr == unique_element)), size=n, replace=False)

    # Replace values at selected indices with unique element
    arr[indices_to_replace] = unique_element

    return arr
    
def shape_imposer(X,Y,S,l_a,l_b,POS):
    D_x = []
    D_y = []
    for x,y in zip(X,Y):
        #set equal no of samples as of pos randomly
        shape_selection = np.random.randint(0,len(S),len((POS)))
        shapes = [S[i] for i in shape_selection]
        L_a = [l_a[j] for j in shape_selection]
        L_b = [l_b[k] for k in shape_selection]
        # Now that the length of L_a,L_b, shapes and pos are same, we can use for loop for data generation
        for s, la, lb, po in zip(shapes,L_a,L_b,POS):
            pks, _ = find_peaks(s)
            if len(pks)!=0:
                b = len(s[:pks[0]])
                a = len(s[pks[0]:])
                #print(s)
                #print(lb)
                #print(po)
                x[po-b:po]+=lb
                x[po:po+a]+=la
                y[po]+=1
            elif len(pks)==0:
                b = 0
                a = len(s)
                x[po:po+len(s)]+=s
                y[po-b+1]+=1            
        D_x.append(x)
        D_y.append(y)
    return D_x,D_y

def controlled_mc(shape_file, required_examples, sample_size, indice_to_add):
    with open(shape_file, 'rb') as f:
        S = pickle.load(f)
    l_a = []
    l_b = []
    for i in S:
        sh = i
        pk,_ = find_peaks(i)
        if len(pk)==0:
            peak = 0
            la = len(i)
            lb = 0
        else:
            peak = pk[0]
            lb = i[:peak]
            la = i[peak:]
        l_a.append(la)
        l_b.append(lb)
    X = np.zeros((required_examples, sample_size))
    Y = np.zeros((required_examples, sample_size))
    #pos = [85,90,95,100,105,110,115,120]
    x, y = shape_imposer(X,Y,S,l_a,l_b,indice_to_add)
    return X, Y