from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from src.model_architectures.unet_models import MODELS
import glob


# Load data
folder_name = 'sh06x2'
train_exe = 50
train_x = np.load(f'./DATASETS/{folder_name}/lab_ch0_shapes_train_x_m_{train_exe}.npy')
train_y = np.load(f'./DATASETS/{folder_name}/lab_ch0_shapes_train_y_m_{train_exe}.npy')

val_x = np.load(f'./DATASETS/{folder_name}/lab_ch0_shapes_val_x_m_{train_exe}.npy')
val_y = np.load(f'./DATASETS/{folder_name}/lab_ch0_shapes_val_y_m_{train_exe}.npy')

test_x = np.load(f'./DATASETS/{folder_name}/lab_ch0_shapes_test_x_m_{train_exe}.npy')
test_y = np.load(f'./DATASETS/{folder_name}/lab_ch0_shapes_test_y_m_{train_exe}.npy')

print(train_x.shape,val_x.shape,test_x.shape)
print(train_y.shape,val_y.shape,test_y.shape)


# Select model and define parameters
bins = 256
channel = 1
loss_func = 'mse'
lr = '1e-4'
opt = tf.keras.optimizers.Nadam(learning_rate=float(lr))
metric = 'mae'
reg = '0'
filt_lst = [64,128,256,512]
dns = 256
drop = '0'
batch_size = 256#
ep=10

# vgg16, unet, resUnet, att_resUnet, unet++
get_model = MODELS()
model_type="Unet10"

if model_type=="attention_resunet":
    FILTER_NUM = 16
    FILTER_SIZE = 3
    NUM_CLASSES = 1
    dropout_rate = 0.2
    batch_norm = False
    model = get_model.Attention_ResUNet(bins,channel,loss_func,opt,metric,FILTER_NUM=FILTER_NUM,batch_norm = False, dropout_rate = float(drop))
elif model_type=="unetpp":
    nb_filter = [16, 32, 64, 128, 256]
    deep_supervision = False
    model = get_model.Nest_Net(bins,channel,loss_func,opt,metric,nb_filter=nb_filter,deep_supervision=deep_supervision)
elif model_type=="Unet10":
    filt_num = 10
    model = get_model.UNET(bins, channel, loss_func, opt, metric, float(reg), filt_num)
elif model_type=="vgg16":
    model = get_model.VGG16(filt_lst,dns,bins,channel,loss_func,opt,metric)
elif model_type=="resUnet":
    model = get_model.resUnet(bins, channel, loss_func, opt, metric, float(reg))
elif model_type=="unetPP48163264":
    nb_filter = [4,8,16,32,64]
    filters = [4,8,16,32,64]
    model = get_model.unetPP(bins,channel,loss_func,opt,metric,float(reg), nb_filter, filters)


# incase loaded data is not in proper shape, reshape it
if len(train_x.shape)<3:
    train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
    val_X = val_x.reshape(val_x.shape[0],val_x.shape[1],1)
    test_X = test_x.reshape(test_x.shape[0],test_x.shape[1],1)
    
    train_y = train_y.reshape(train_y.shape[0],train_y.shape[1],1)
    val_y = val_y.reshape(val_y.shape[0],val_y.shape[1],1)
    test_y = test_y.reshape(test_y.shape[0],test_y.shape[1],1)

    
# Select folder and define callbacks for training model
folder_name = 'sh06x2'
chnl = 'ch0'
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)
mc = tf.keras.callbacks.ModelCheckpoint(f'./models/{folder_name}/{model_type}_{folder_name}_{chnl}__256_m_1280_{loss_func}_lr_{lr}_{ep}_DROPOUT{reg}.h5',
                                       monitor='val_loss', model='min', verbose=1, save_best_only=True)


# Train the model
history=model.fit(train_X[:],train_y[:],
                batch_size=batch_size,epochs=ep,
                validation_data=(val_X[:],val_y[:]),shuffle=True, callbacks=[callback, mc])



# Save the training perormance to selected folder in the form of .npy
np.save(f'./model_performance/{folder_name}/loss_{model_type}_{folder_name}_{chnl}__256_m_1280_{loss_func}_lr_{lr}_{ep}_DROPOUT{reg}.npy',
       history.history['loss'])
np.save(f'./model_performance/{folder_name}/val_loss_{model_type}_{folder_name}_{chnl}__256_m_1280_{loss_func}_lr_{lr}_{ep}_DROPOUT{reg}.npy',
       history.history['val_loss'])

np.save(f'./model_performance/{folder_name}/mae_{model_type}_{folder_name}_{chnl}__256_m_1280_{loss_func}_lr_{lr}_{ep}_DROPOUT{reg}.npy',
       history.history['mae'])
np.save(f'./model_performance/{folder_name}/val_mae_{model_type}_{folder_name}_{chnl}__256_m_1280_{loss_func}_lr_{lr}_{ep}_DROPOUT{reg}.npy',
       history.history['val_mae'])