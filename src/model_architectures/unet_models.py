import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class MODELS:
    
    def __init__(self):
        pass

    def VGG16(self, filt_lst,dns,bins,channel,loss_func,opt,metric):
        #filt_lst = [64,128,256,512]
        inputs = tf.keras.layers.Input((bins, channel))

        x = tf.keras.layers.Conv1D(filt_lst[0],3,padding="same",activation='relu')(inputs)
        x = tf.keras.layers.Conv1D(filt_lst[0],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(2,strides=2)(x)

        x = tf.keras.layers.Conv1D(filt_lst[1],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.Conv1D(filt_lst[1],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(2,strides=2)(x)

        x = tf.keras.layers.Conv1D(filt_lst[2],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.Conv1D(filt_lst[2],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.Conv1D(filt_lst[2],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(2,strides=2)(x)

        x = tf.keras.layers.Conv1D(filt_lst[3],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.Conv1D(filt_lst[3],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.Conv1D(filt_lst[3],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(2,strides=2)(x)

        x = tf.keras.layers.Conv1D(filt_lst[3],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.Conv1D(filt_lst[3],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.Conv1D(filt_lst[3],3,padding="same",activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(2,strides=2)(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(dns,activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(dns,activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(256,activation='relu')(x)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
        model.summary()
        return model
        
    def UNET(self, bins, channel, loss_func, opt, metric, reg, filt_num):
        # Build the model

        inputs = tf.keras.layers.Input((bins,channel))
        c1 = tf.keras.layers.Conv1D(filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(inputs)#
        #c1 = tf.keras.layers.Dropout(reg)(c1)
        c1 = tf.keras.layers.Conv1D(filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(c1)#(s)                       
        p1 = tf.keras.layers.MaxPooling1D(2)(c1)
        c2 = tf.keras.layers.Conv1D(2*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(p1)#(s)
        c2 = tf.keras.layers.Dropout(reg)(c2)
        c2 = tf.keras.layers.Conv1D(2*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(c2)#(s)                           
        p2 = tf.keras.layers.MaxPooling1D(2)(c2)


        c3 = tf.keras.layers.Conv1D(4*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(p2)#(s)                           
        c3 = tf.keras.layers.Dropout(reg)(c3)
        c3 = tf.keras.layers.Conv1D(4*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(c3)#(s)                           
        p3 = tf.keras.layers.MaxPooling1D(2)(c3)
        c4 = tf.keras.layers.Conv1D(8*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(p3)#(s)                           
        c4 = tf.keras.layers.Dropout(reg)(c4)
        c4 = tf.keras.layers.Conv1D(8*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(c4)#(s)                           
        p4 = tf.keras.layers.MaxPooling1D(2)(c4)

        c5 = tf.keras.layers.Conv1D(16*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(p4)#(s)                           
        c5 = tf.keras.layers.Dropout(reg)(c5)
        c5 = tf.keras.layers.Conv1D(16*filt_num,3,activation='relu', kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(c5)#(s)                           
        p5 = tf.keras.layers.MaxPooling1D(2)(c5)



        u6 = tf.keras.layers.Conv1DTranspose(8*filt_num,2,strides=2,padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6,c4])
        c6 = tf.keras.layers.Conv1D(8*filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u6)
        c6 = tf.keras.layers.Dropout(reg)(c6)
        c6 = tf.keras.layers.Conv1D(8*filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u6)


        u7 = tf.keras.layers.Conv1DTranspose(4*filt_num,2,strides=2,padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7,c3])
        u6 = tf.keras.layers.Reshape((u6.shape[1],u6.shape[2],1), input_shape=u6.shape)
        c7 = tf.keras.layers.Conv1D(4*filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u7)
        c7 = tf.keras.layers.Dropout(reg)(c7)
        c7 = tf.keras.layers.Conv1D(4*filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u7)


        u8 = tf.keras.layers.Conv1DTranspose(2*filt_num,2,strides=2,padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8,c2])
        c8 = tf.keras.layers.Conv1D(2*filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u8)
        c8 = tf.keras.layers.Dropout(reg)(c8)
        c8 = tf.keras.layers.Conv1D(2*filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u8)


        u9 = tf.keras.layers.Conv1DTranspose(filt_num,2,strides=2,padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9,c1])
        c9 = tf.keras.layers.Conv1D(filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u9)
        #c9 = tf.keras.layers.Dropout(reg)(c9)
        c9 = tf.keras.layers.Conv1D(filt_num,3,activation='relu',kernel_initializer='he_normal', #kernel_regularizer=reg,
        padding='same')(u9)
        outputs = tf.keras.layers.Conv1D(1,1,activation='relu')(c9)
        print("layer output: {}".format(outputs.shape))



        def rmse(y_true, y_pred):
                return K.sqrt(K.mean(K.square(y_pred - y_true))) 

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        #model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.00001), loss=loss_func, metrics=['mse'])
        model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
        model.summary()

        return model
    
    def convolutional_block(self, X, f, filters, stage, block, s = 2):
    
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_conv_branch_'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path 
        X = tf.keras.layers.Conv1D(F1, 1, activation='relu', strides = s, name = conv_name_base + 'conv_0')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
        #X = BatchNormalization(name = bn_name_base + '2a')(X)
        X = tf.keras.layers.Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = tf.keras.layers.Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Third component of main path (≈2 lines)
        X = tf.keras.layers.Conv1D(filters = F3, kernel_size = 1, activation='relu', strides = 1, padding = 'valid', name = conv_name_base + 'conv_1')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = tf.keras.layers.Conv1D(filters = F3, activation='relu', kernel_size = 1, strides = s, padding = 'valid', name = conv_name_base + 'conv_shortcut')(X_shortcut)#,
                            #kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation('relu')(X)
        return X
    
    def identity_block(self, X, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch_'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X   
        # First component of main path
        X = tf.keras.layers.Conv1D(filters = F1, kernel_size = 1, activation='relu', strides = 1, padding = 'valid', name = conv_name_base + 'identity_0')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
        X = tf.keras.layers.Activation('relu')(X)   
        # Second component of main path (≈3 lines)
        X = tf.keras.layers.Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Third component of main path (≈2 lines)
        X = tf.keras.layers.Conv1D(filters = F3, kernel_size = 1, activation='relu', strides = 1, padding = 'valid', name = conv_name_base + 'identity_1')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation('relu')(X)
        return X
    
    def resUnet(self, bins, channel, loss_func, opt, MeTric, reg):
        inputs = tf.keras.layers.Input((bins, channel))
        X = tf.keras.layers.Conv1D(16, 3, strides=1, name='conv1',padding='same')(inputs)
        c1 = self.convolutional_block(X, f=3, filters=[2, 8, 16], stage=2, block='a000', s=1)
        c1 = tf.keras.layers.Dropout(reg)(c1)
        c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-1')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-2')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-3')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-5')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-6')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-7')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-8')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-9')
        #c1 = self.identity_block(c1, 3, [2, 8, 16], stage=2, block='ab1-10')
        p1 = tf.keras.layers.MaxPooling1D(2)(c1)

        print('-----------------------------------------------------------------------------')
        c2 = self.convolutional_block(p1, f=3, filters=[4, 8, 32], stage=2, block='a001', s=1)
        c2 = tf.keras.layers.Dropout(reg)(c2)
        c2 = self.identity_block(c2, 3, [4, 8, 32], stage=2, block='ab2')
        #c2 = self.identity_block(c2, 3, [4, 8, 32], stage=2, block='ab3')
        #c2 = self.identity_block(c2, 3, [4, 8, 32], stage=2, block='ab3-1')
        #c2 = self.identity_block(c2, 3, [4, 8, 32], stage=2, block='ab3-2')
        #c2 = self.identity_block(c2, 3, [4, 8, 32], stage=2, block='ab3-3')
        #c2 = self.identity_block(c2, 3, [4, 8, 32], stage=2, block='ab3-4')
        p2 = tf.keras.layers.MaxPooling1D(2)(c2)

        print('-----------------------------------------------------------------------------')
        c3 = self.convolutional_block(p2, f=3, filters=[8, 8, 64], stage=2, block='a002', s=1)
        c3 = tf.keras.layers.Dropout(reg)(c3)
        c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab4')
        #c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab5')
        #c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab5-1')
        #c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab5-2')
        #c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab5-3')
        #c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab5-4')
        #c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab5-5')
        #c3 = self.identity_block(c3, 3, [8, 8, 64], stage=2, block='ab5-6')
        p3 = tf.keras.layers.MaxPooling1D(2)(c3)

        print('-----------------------------------------------------------------------------')
        c4 = self.convolutional_block(p3, f=3, filters=[16, 16, 128], stage=2, block='a003', s=1)
        c4 = tf.keras.layers.Dropout(reg)(c4)
        c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab6')
        #c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab7')
        #c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab7-1')
        #c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab7-2')
        #c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab7-3')
        #c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab7-4')
        #c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab7-5')
        #c4 = self.identity_block(c4, 3, [16, 16, 128], stage=2, block='ab7-6')
        p4 = tf.keras.layers.MaxPooling1D(2)(c4)

        print('-----------------------------------------------------------------------------')
        c5 = self.convolutional_block(p4, f=3, filters=[32, 32, 256], stage=2, block='a004', s=1)
        c5 = tf.keras.layers.Dropout(reg)(c5)
        c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab8')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-1')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-2')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-3')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-4')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-5')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-6')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-7')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab8-8')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-9')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-10')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-11')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-12')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-13')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-14')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-15')
        #c5 = self.identity_block(c5, 3, [32, 32, 256], stage=2, block='ab9-16')
        p5 = tf.keras.layers.MaxPooling1D(2)(c5)
        
        print('-----------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------')
        
        u6 = tf.keras.layers.Conv1DTranspose(128,2,strides=2,padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6,c4])
        c6 = self.convolutional_block(u6, f=3, filters=[16, 16, 128], stage=2, block='a111', s=1)
        c6 = tf.keras.layers.Dropout(reg)(c6)
        c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112')
        #c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112-1')
        #c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112-2')
        #c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112-3')
        #c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112-4')
        #c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112-5')
        #c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112-6')
        #c6 = self.identity_block(c6, 3, [16, 16, 128], stage=2, block='ab112-7')
        print('-----------------------------------------------------------')


        u7 = tf.keras.layers.Conv1DTranspose(64,2,strides=2,padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7,c3])
        c7 = self.convolutional_block(u7, f=3, filters=[8, 8, 64], stage=2, block='a113', s=1)
        c7 = tf.keras.layers.Dropout(reg)(c7)
        c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114')
        #c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114-1')
        #c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114-2')
        #c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114-3')
        #c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114-4')
        #c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114-5')
        #c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114-6')
        #c7 = self.identity_block(c7, 3, [8, 8, 64], stage=2, block='ab114-7')
        print('-----------------------------------------------------------')

        u8 = tf.keras.layers.Conv1DTranspose(32,2,strides=2,padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8,c2])
        c8 = self.convolutional_block(u8, f=3, filters=[4, 8, 32], stage=2, block='a115', s=1)
        c8 = tf.keras.layers.Dropout(reg)(c8)
        c8 = self.identity_block(c8, 3, [4, 8, 32], stage=2, block='ab116')
        #c8 = self.identity_block(c8, 3, [4, 8, 32], stage=2, block='ab116-1')
        #c8 = self.identity_block(c8, 3, [4, 8, 32], stage=2, block='ab116-2')
        #c8 = self.identity_block(c8, 3, [4, 8, 32], stage=2, block='ab116-3')
        #c8 = self.identity_block(c8, 3, [4, 8, 32], stage=2, block='ab116-4')
        #c8 = self.identity_block(c8, 3, [4, 8, 32], stage=2, block='ab116-5')
        print('-------------------------------------------------------------')

        u9 = tf.keras.layers.Conv1DTranspose(16,2,strides=2,padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9,c1])
        #u6 = tf.keras.layers.Reshape((u6.shape[1],u6.shape[2],1), input_shape=u6.shape)
        c9 = self.convolutional_block(u9, f=3, filters=[2, 8, 16], stage=2, block='a117', s=1)
        c9 = tf.keras.layers.Dropout(reg)(c9)
        c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-1')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-2')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-3')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-4')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-5')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-6')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-7')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-8')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-9')
        #c9 = self.identity_block(c9, 3, [2, 8, 16], stage=2, block='ab118-10')
        outputs = tf.keras.layers.Conv1D(1,1,activation='relu')(c9)

        def rmse(y_true, y_pred):
                return K.sqrt(K.mean(K.square(y_pred - y_true))) 
        #opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=opt, loss=loss_func,metrics=[MeTric])
        model.summary()

        return model
    ###############################################################
    def conv_block(self, x, filter_size, size, dropout, batch_norm=False):
            conv = layers.Conv1D(size, filter_size, padding="same")(x)
            if batch_norm is True:
                conv = layers.BatchNormalization()(conv)
            conv = layers.Activation("relu")(conv)
            conv = layers.Conv1D(size, filter_size, padding="same")(conv)
            if batch_norm is True:
                conv = layers.BatchNormalization()(conv)
            conv = layers.Activation("relu")(conv)

            if dropout > 0:
                conv = layers.Dropout(dropout)(conv)

            return conv

    def res_conv_block(self, x, filter_size, size, dropout, batch_norm=False):
        conv = layers.Conv1D(size, filter_size, padding="same")(x)
        if batch_norm is True:
            conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("relu")(conv)

        conv = layers.Conv1D(size, filter_size, padding="same")(x)
        if batch_norm is True:
            conv = layers.BatchNormalization()(conv)
        #conv = layers.Activation("relu")(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv1D(size, kernel_size=1, padding="same")(x)
        if batch_norm is True:
            conv = layers.BatchNormalization()(shortcut)

        res_path = layers.add([shortcut, conv])
        res_path = layers.Activation("relu")(res_path)
        return res_path

    def repeat_elem(self, tensor, rep):
        return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=2), arguments={'repnum': rep})(tensor)

    def gating_signal(self, input, out_size, batch_norm=False):
        x = layers.Conv1D(out_size, 1, padding="same")(input)
        if batch_norm is True:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def attention_block(self, x, gating, inter_shape):
        shape_x = K.int_shape(x)
        shape_g = K.int_shape(gating)

    # Getting the x signal to the same shape as the gating signal
        theta_x = layers.Conv1D(inter_shape, 2, strides= 2, padding='same')(x)  # 16
        shape_theta_x = K.int_shape(theta_x)

    # Getting the gating signal to the same number of filters as the inter_shape
        phi_g = layers.Conv1D(inter_shape, 1, padding='same')(gating)
        upsample_g = layers.Conv1DTranspose(inter_shape, 3,
                                    strides=shape_theta_x[2] // shape_g[2],
                                    padding='same')(phi_g)  # 16

        concat_xg = layers.add([upsample_g, theta_x])
        act_xg = layers.Activation('relu')(concat_xg)
        psi = layers.Conv1D(1, 1, padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = layers.UpSampling1D(size=(shape_x[1] // shape_sigmoid[1]))(sigmoid_xg)  # 32

        upsample_psi = self.repeat_elem(upsample_psi, shape_x[2])

        y = layers.multiply([upsample_psi, x])

        result = layers.Conv1D(shape_x[2], 1, padding='same')(y)
        result = layers.Conv1D(1, 1, padding='same')(y)
        result_bn = layers.BatchNormalization()(result)
        return result_bn

    def UNet(self, input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
        FILTER_NUM = 64
        FILTER_SIZE = 3
        UP_SAMP_SIZE = 2

        inputs = layers.Input(input_shape, dtype=tf.float32)

        conv_128 = self.conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling1D(pool_size=2)(conv_128)

        conv_64 = self.conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        pool_32 = layers.MaxPooling1D(pool_size=2)(conv_64)

        conv_32 = self.conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling1D(pool_size=2)(conv_32)

        conv_16 = self.conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling1D(pool_size=2)(conv_16)

        conv_8 = self.conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

        # Upsampling layers

        up_16 = layers.UpSampling1D(size=UP_SAMP_SIZE)(conv_8)
        up_16 = layers.concatenate([up_16, conv_16])
        up_conv_16 = self.conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 7

        up_32 = layers.UpSampling1D(size=UP_SAMP_SIZE)(up_conv_16)
        up_32 = layers.concatenate([up_32, conv_32])
        up_conv_32 = self.conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 8

        up_64 = layers.UpSampling1D(size=UP_SAMP_SIZE)(up_conv_32)
        up_64 = layers.concatenate([up_64, conv_64])
        up_conv_64 = self.conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 9

        up_128 = layers.UpSampling1D(size=UP_SAMP_SIZE)(up_conv_64)
        up_128 = layers.concatenate([up_128, conv_128])
        up_conv_128 = self.conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

        # 1*1 convolutional layers

        conv_final = layers.Conv1D(NUM_CLASSES, kernel_size=1)(up_conv_128)
        conv_final = layers.BatchNormalization()(conv_final)
        conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

        # Model 
        model = models.Model(inputs, conv_final, name="UNet")
        print(model.summary())
        return

    def Attention_ResUNet(self, bins, channel, loss_func, opt, metric, FILTER_NUM=64, FILTER_SIZE=3, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
        '''
        Rsidual UNet, with attention 

        '''
        # network structure
        #FILTER_NUM = 64 # number of basic filters for the first layer
        #FILTER_SIZE = 3 # size of the convolutional filter
        UP_SAMP_SIZE = 2 # size of upsampling filters
        # input data
        # dimension of the image depth
        inputs = layers.Input((bins,channel))
        #inputs = layers.Input(input_shape, dtype=tf.float32)
        axis = 2

        # Downsampling layers
        # DownRes 1, double residual convolution + pooling
        conv_128 = self.res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling1D(pool_size=2)(conv_128)
        # DownRes 2
        conv_64 = self.res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        pool_32 = layers.MaxPooling1D(pool_size=2)(conv_64)
        # DownRes 3
        conv_32 = self.res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling1D(pool_size=2)(conv_32)
        # DownRes 4
        conv_16 = self.res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling1D(pool_size=2)(conv_16)
        # DownRes 5, convolution only
        conv_8 = self.res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        gating_16 = self.gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
        att_16 = self.attention_block(conv_16, gating_16, 8*FILTER_NUM)
        up_16 = layers.UpSampling1D(size=UP_SAMP_SIZE)(conv_8)
        print(up_16.shape,att_16.shape)
        up_16 = layers.concatenate([up_16, att_16], axis=axis)
        up_conv_16 = self.res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 7
        gating_32 = self.gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
        att_32 = self.attention_block(conv_32, gating_32, 4*FILTER_NUM)
        up_32 = layers.UpSampling1D(size=UP_SAMP_SIZE)(up_conv_16)
        up_32 = layers.concatenate([up_32, att_32], axis=axis)
        up_conv_32 = self.res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 8
        gating_64 = self.gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
        att_64 = self.attention_block(conv_64, gating_64, 2*FILTER_NUM)
        up_64 = layers.UpSampling1D(size=UP_SAMP_SIZE)(up_conv_32)
        up_64 = layers.concatenate([up_64, att_64], axis=axis)
        up_conv_64 = self.res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 9
        gating_128 = self.gating_signal(up_conv_64, FILTER_NUM, batch_norm)
        att_128 = self.attention_block(conv_128, gating_128, FILTER_NUM)
        up_128 = layers.UpSampling1D(size=UP_SAMP_SIZE)(up_conv_64)
        up_128 = layers.concatenate([up_128, att_128], axis=axis)
        up_conv_128 = self.res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

        # 1*1 convolutional layers

        conv_final = layers.Conv1D(NUM_CLASSES, kernel_size=1)(up_conv_128)
        #conv_final = layers.BatchNormalization(axis=axis)(conv_final)
        if batch_norm is True:
            conv_final = layers.BatchNormalization()(conv_final)
        conv_final = layers.Activation('relu')(conv_final)  #Change to softmax for multichannel

        # Model integration
        model = models.Model(inputs, conv_final, name="AttentionResUNet")
        model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
        model.summary()
        return model

        
    def standard_unit(self, input_tensor, stage, nb_filter, kernel_size=3):

        act = 'elu'
        dropout_rate =0.2
        x = layers.Conv1D(nb_filter, kernel_size, activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.L2(1e-4))(input_tensor)
        x = layers.Dropout(dropout_rate, name='dp'+stage+'_1')(x)
        x = layers.Conv1D(nb_filter, kernel_size, activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
        x = layers.Dropout(dropout_rate, name='dp'+stage+'_2')(x)

        return x
    
    def Nest_Net(self, bins, channel, loss_func, opt, metric, nb_filter = [32,64,128,256,512], num_class=1, deep_supervision=False):

        
        act = 'relu'

        bn_axis = 2
        img_input = layers.Input(shape=(bins, channel), name='main_input')

        conv1_1 = self.standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
        pool1 = layers.MaxPool1D(2, strides=2, name='pool1')(conv1_1)

        conv2_1 = self.standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
        pool2 = layers.MaxPool1D(2, strides=2, name='pool2')(conv2_1)

        up1_2 = layers.Conv1DTranspose(nb_filter[0], 2, strides=2, name='up12', padding='same')(conv2_1)
        conv1_2 = layers.concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
        conv1_2 = self.standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

        conv3_1 = self.standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
        pool3 = layers.MaxPool1D(2, strides=2, name='pool3')(conv3_1)

        up2_2 = layers.Conv1DTranspose(nb_filter[1], 2, strides=2, name='up22', padding='same')(conv3_1)
        conv2_2 = layers.concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
        conv2_2 = self.standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

        up1_3 = layers.Conv1DTranspose(nb_filter[0], 2, strides=2, name='up13', padding='same')(conv2_2)
        conv1_3 = layers.concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
        conv1_3 = self.standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

        conv4_1 = self.standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
        pool4 = layers.MaxPool1D(2, strides=2, name='pool4')(conv4_1)

        up3_2 = layers.Conv1DTranspose(nb_filter[2], 2, strides=2, name='up32', padding='same')(conv4_1)
        conv3_2 = layers.concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
        conv3_2 = self.standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

        up2_3 = layers.Conv1DTranspose(nb_filter[1], 2, strides=2, name='up23', padding='same')(conv3_2)
        conv2_3 = layers.concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
        conv2_3 = self.standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

        up1_4 = layers.Conv1DTranspose(nb_filter[0], 2, strides=2, name='up14', padding='same')(conv2_3)
        conv1_4 = layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
        conv1_4 = self.standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

        conv5_1 = self.standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

        up4_2 = layers.Conv1DTranspose(nb_filter[3], 2, strides=2, name='up42', padding='same')(conv5_1)
        conv4_2 = layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
        conv4_2 = self.standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

        up3_3 = layers.Conv1DTranspose(nb_filter[2], 2, strides=2, name='up33', padding='same')(conv4_2)
        conv3_3 = layers.concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
        conv3_3 = self.standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

        up2_4 = layers.Conv1DTranspose(nb_filter[1], 2, strides=2, name='up24', padding='same')(conv3_3)
        conv2_4 = layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
        conv2_4 = self.standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

        up1_5 = layers.Conv1DTranspose(nb_filter[0], 2, strides=2, name='up15', padding='same')(conv2_4)
        conv1_5 = layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
        conv1_5 = self.standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

        nestnet_output_1 = layers.Conv1D(num_class, 1, activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv1_2)
        nestnet_output_2 = layers.Conv1D(num_class, 1, activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv1_3)
        nestnet_output_3 = layers.Conv1D(num_class, 1, activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv1_4)
        nestnet_output_4 = layers.Conv1D(num_class, 1, activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv1_5)

        if deep_supervision:
            model = models.Model(img_input, [nestnet_output_1,nestnet_output_2,nestnet_output_3,nestnet_output_4])
        else:
            model = models.Model(img_input, [nestnet_output_4])
        model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
        model.summary()

        return model
    
    def unetPP(self,bins,channel,loss_func,opt,metric,reg, nb_filter, filters):
        inputs = Input((bins, channel))
        #s = Lambda(lambda x: x / 255) (inputs)

        c1 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
        c1 = Dropout(reg) (c1)
        c1 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (c1)
        c1 = Dropout(reg) (c1)
        p1 = MaxPooling1D(2, strides=2) (c1)

        c2 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(reg) (c2)
        c2 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (c2)
        c2 = Dropout(reg) (c2)
        p2 = MaxPooling1D(2, strides=2) (c2)

        up1_2 = Conv1DTranspose(nb_filter[0], 2, strides=2, name='up12', padding='same')(c2)
        conv1_2 = concatenate([up1_2, c1], name='merge12')
        c3 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv1_2)
        c3 = Dropout(reg) (c3)
        c3 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (c3)
        c3 = Dropout(reg) (c3)

        conv3_1 = Conv1D(filters[2], 3, activation='relu', kernel_initializer='he_normal', padding='same') (p2)
        conv3_1 = Dropout(reg) (conv3_1)
        conv3_1 = Conv1D(filters[2], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv3_1)
        conv3_1 = Dropout(reg) (conv3_1)
        pool3 = MaxPooling1D(2, strides=2, name='pool3')(conv3_1)

        up2_2 = Conv1DTranspose(nb_filter[1], 2, strides=2, name='up22', padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, c2], name='merge22')#, axis=3) #x10
        conv2_2 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv2_2)
        conv2_2 = Dropout(reg) (conv2_2)
        conv2_2 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv2_2)
        conv2_2 = Dropout(reg) (conv2_2)

        up1_3 = Conv1DTranspose(nb_filter[0], 2, strides=2, name='up13', padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, c1, c3], name='merge13')#, axis=3)
        conv1_3 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv1_3)
        conv1_3 = Dropout(reg) (conv1_3)
        conv1_3 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv1_3)
        conv1_3 = Dropout(reg) (conv1_3)

        conv4_1 = Conv1D(filters[3], 3, activation='relu', kernel_initializer='he_normal', padding='same') (pool3)
        conv4_1 = Dropout(reg) (conv4_1)
        conv4_1 = Conv1D(filters[3], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv4_1)
        conv4_1 = Dropout(reg) (conv4_1)
        pool4 = MaxPooling1D(2, strides=2, name='pool4')(conv4_1)

        up3_2 = Conv1DTranspose(nb_filter[2], 2, strides=2, name='up32', padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], name='merge32')#, axis=3) #x20
        conv3_2 = Conv1D(filters[2], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv3_2)
        conv3_2 = Dropout(reg) (conv3_2)
        conv3_2 = Conv1D(filters[2], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv3_2)
        conv3_2 = Dropout(reg) (conv3_2)

        up2_3 = Conv1DTranspose(nb_filter[1], 2, strides=2, name='up23', padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, c2, conv2_2], name='merge23')#, axis=3)
        conv2_3 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv2_3)
        conv2_3 = Dropout(reg) (conv2_3)
        conv2_3 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv2_3)
        conv2_3 = Dropout(reg) (conv2_3)

        up1_4 = Conv1DTranspose(nb_filter[0], 2, strides=2, name='up14', padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, c1, c3, conv1_3], name='merge14')#, axis=3)
        conv1_4 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv1_4)
        conv1_4 = Dropout(reg) (conv1_4)
        conv1_4 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv1_4)
        conv1_4 = Dropout(reg) (conv1_4)

        conv5_1 = Conv1D(filters[4], 3, activation='relu', kernel_initializer='he_normal', padding='same') (pool4)
        conv5_1 = Dropout(reg) (conv5_1)
        conv5_1 = Conv1D(filters[4], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv5_1)
        conv5_1 = Dropout(reg) (conv5_1)

        up4_2 = Conv1DTranspose(nb_filter[3], 2, strides=2, name='up42', padding='same')(conv5_1)
        conv4_2 = concatenate([up4_2, conv4_1], name='merge42')#, axis=3) #x30
        conv4_2 = Conv1D(filters[3], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv4_2)
        conv4_2 = Dropout(reg) (conv4_2)
        conv4_2 = Conv1D(filters[3], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv4_2)
        conv4_2 = Dropout(reg) (conv4_2)

        up3_3 = Conv1DTranspose(nb_filter[2], 2, strides=2, name='up33', padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33')#, axis=3)
        conv3_3 = Conv1D(filters[2], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv3_3)
        conv3_3 = Dropout(reg) (conv3_3)
        conv3_3 = Conv1D(filters[2], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv3_3)
        conv3_3 = Dropout(reg) (conv3_3)

        up2_4 = Conv1DTranspose(nb_filter[1], 2, strides=2, name='up24', padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, c2, conv2_2, conv2_3], name='merge24')#, axis=3)
        conv2_4 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv2_4)
        conv2_4 = Dropout(reg) (conv2_4)
        conv2_4 = Conv1D(filters[1], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv2_4)
        conv2_4 = Dropout(reg) (conv2_4)

        up1_5 = Conv1DTranspose(nb_filter[0], 2, strides=2, name='up15', padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, c1, c3, conv1_3, conv1_4], name='merge15')#, axis=3)
        conv1_5 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv1_5)
        conv1_5 = Dropout(reg) (conv1_5)
        conv1_5 = Conv1D(filters[0], 3, activation='relu', kernel_initializer='he_normal', padding='same') (conv1_5)
        conv1_5 = Dropout(reg) (conv1_5)

        nestnet_output_4 = Conv1D(1, 1, activation='relu', kernel_initializer = 'he_normal',  name='output_4', padding='same')(conv1_5)

        model = Model([inputs], [nestnet_output_4])
        model.compile(optimizer=opt, loss=loss_func, metrics=[metric])

        #tf.keras.utils.plot_model(model, to_file='model_test.png', show_shapes=True, show_layer_names=True)
        model.summary()
        return model


        
"""bins = 256
channel = 1
loss_func = 'mae'
opt = tf.keras.optimizers.Nadam(learning_rate=0.0001)
MeTric = 'mse'
reg = tf.keras.regularizers.L2(0.000000000000003)

model = MODELS().resUnet(bins, channel, loss_func, opt, MeTric)
model.summary()"""