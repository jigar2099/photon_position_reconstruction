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
import glob
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn import metrics





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


def prediction_error(ch0_path, ch0_label_path, model_dir_path_form_glob, examples):
    for i in glob.glob(model_dir_path_form_glob)[:]:
        print('*'*50)
        print(i)
        print('*'*50)

        test_x = np.load(ch0_path)
        test_y = np.load(ch0_label_path)

        model = keras.models.load_model(i,compile=False)
        model.compile()
        predictions = model.predict(test_x)

        #import matplotlib.pyplot as plt
        #import numpy as np
        from matplotlib import gridspec
        
        samp_ind = np.random.randint(examples, test_x.shape[0],1)[0]#--->268mhz

        fig = plt.figure(figsize=(17,5))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
        # the first subplot
        ax0 = plt.subplot(gs[0])
        # log scale for axis Y of the first subplot
        #ax0.set_yscale("log")
        line0, = ax0.plot(test_x[samp_ind], color='black')
        ax0.set_ylabel('Counts')
        # the second subplot
        # shared axis X
        ax1 = plt.subplot(gs[1], sharex = ax0)
        line1, = ax1.plot(test_y[samp_ind], color='b', marker='o', linestyle='-')
        line2, = ax1.plot(predictions[samp_ind], color='red', marker='.', linestyle='-', alpha=0.9)
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1.set_ylabel('Photons')
        # remove last tick label for the second subplot
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        # put legend on first subplot
        ax0.legend((line0, line1, line2), ('Data sample', 'Label vector', 'Predictions'), loc='upper left')
        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)
        ax0.set_title('Ground truth({}MHz) and prediction({}MHz) comparison'.format(np.round((1000/(1.6*256))*np.sum(test_y[samp_ind]),2), np.round((1000/(1.6*256))*np.sum(predictions[samp_ind]),2)))
        ax1.set_xlabel('Time-step (1.6ns)')
        #plt.savefig('Label({}MHz)-pred_comp({}MHz)'.format(np.int((1000/(1.6*256))*np.sum(test_y[samp_ind])), np.int((1000/(1.6*256))*np.sum(predictions[samp_ind]))))
        plt.show()

        avg_true_rate = []
        avg_pred_rate = []
        std_pred_rate = []

        flat_predictions = []
        flat_labels = []

        cluster_pred = []
        cluster_labels=[]

        predictions1 = predictions.reshape(predictions.shape[0],predictions.shape[1])
        test_y1 = test_y.reshape(test_y.shape[0],test_y.shape[1])
        ref_arr = np.arange(predictions.shape[0])[::int(predictions.shape[0]/predictions.shape[1])]
        for i in range(1,len(ref_arr))[:]:
            low_lim = ref_arr[i-1]
            up_lim = ref_arr[i]

            pred_set = predictions1[low_lim:up_lim]
            label_set = test_y1[low_lim:up_lim]
            flat_pred = pred_set.flatten()
            flat_pred = np.rint(flat_pred)
            flat_pred_original = np.array([iii for iii in flat_pred])
            flat_label = label_set.flatten()

            #print(flat_pred)
            clus_pred = np.diff(flat_pred)
            clus_label = np.diff(flat_label)

            flat_predictions.append(flat_pred)
            flat_labels.append(flat_label)

            cluster_pred.append(clus_pred)
            cluster_labels.append(clus_label)

            #
            true_rate = (1000/(1.6*256))*np.mean(np.sum(label_set,axis=1))
            avg_true_rate.append(true_rate)

            pred_rate = (1000/(1.6*256))*np.mean(np.sum(pred_set,axis=1))
            avg_pred_rate.append(pred_rate)

            pred_rate_std = (1000/(1.6*256))*np.std(np.sum(label_set - pred_set,axis=1))
            std_pred_rate.append(pred_rate_std)

        flat_predictions = np.array(flat_predictions)
        flat_labels = np.array(flat_labels)

        cluster_pred = np.array(cluster_pred)
        cluster_labels = np.array(cluster_labels)

        avg_true_rate = np.array(avg_true_rate)
        avg_pred_rate = np.array(avg_pred_rate)
        std_pred_rate = np.array(std_pred_rate)

        error = avg_true_rate - avg_pred_rate

        import matplotlib as mpl

        # Select a color map
        cmap = mpl.cm.viridis#bwr

        # Some Test data

        x = avg_true_rate # np.linspace(-4, 4, npts) # avg_true_rate
        y = error # norm.pdf(x) # error
        z = std_pred_rate # np.sin(2 * x) # std_pred_rate
        normalize = mpl.colors.Normalize(vmin=z.min(), vmax=z.max())

        # The plot
        fig = plt.figure(figsize=(18,7))
        ax = fig.add_axes([0.12, 0.12, 0.68, 0.78])
        plt.plot(x, y, color="black")
        plt.xlabel('Rate (MHz)')
        plt.ylabel(' Average error (MHz)')
        for i in range(len(x)-1):
            plt.fill_between([x[i], x[i+1]], [y[i], y[i+1]], color=cmap(normalize(z[i])))

        cbax = fig.add_axes([0.82, 0.12, 0.03, 0.78])
        cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=normalize, orientation='vertical')
        cb.set_label("Error uncertainty in a waveform (MHz)", rotation=270, labelpad=15)
        #plt.savefig('shaula_00000-003_ch0_error_ana.png')

def confusion_mat(ch0_path, ch0_label_path, model_dir_path_form_glob):
    for i in glob.glob(model_dir_path_form_glob):
        print(i)
        test_x = np.load(ch0_path)
        test_y = np.load(ch0_label_path)
        model = keras.models.load_model(i,compile=False)
        model.compile()

        predictions = model.predict(test_x)

        lower_lim = 6
        upper_lim = test_y.shape[1]-10

        test_y1 = test_y[:, lower_lim:upper_lim]
        predictions1 = predictions.reshape(predictions.shape[0],predictions.shape[1])[:, lower_lim:upper_lim]
        predictions1 = np.rint(predictions1)# round the predictions
        test_y1 = np.rint(test_y1)# riund the labels


        cm = confusion_matrix(test_y1.ravel(), predictions1.ravel())
        cm = cm/cm.sum(axis=1)[:,None]

        where_are_NaNs = np.isnan(cm)
        cm[where_are_NaNs] = 0
        cm = np.around(cm, decimals=2)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        #cm_display.plot()
        fig, ax = plt.subplots(figsize=(13,10))
        cm_display.plot(ax=ax)
        #plt.savefig('recall_confusion_matrix.png')
        plt.show()
import numpy as np

def remove_nonzero_neighbors(arr):
    # Find the indices of the smallest elements in the array
    smallest_indices = np.where(arr == np.min(arr))[0]

    # Remove the indices of smallest elements whose neighbors are non-zero
    modified_indices = []
    for i in smallest_indices:
        if i > 0 and i < len(arr) - 1:
            if arr[i-1] == 0 and arr[i+1] == 0:
                modified_indices.append(i)

    return modified_indices
