""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""

import os
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam
#from keras.layers.normalization import BatchNormalization
import h5py
import functions as fn
import csv
import gc

env_jobid = os.environ['PBS_JOBID']
pbs_jobid = [int(s) for s in env_jobid.split('.') if s.isdigit()]

#from sklearn.preprocessing import MinMaxScaler

print('Loading data..')
# fix random seed for reproducibility

np.random.seed(7)

# Load Data
#### V2 files ####

input_filename2 = '/home/dlopezmo/datasetsV2/100box/ball_in_square_10ballsv2_tbox100.h5'
output_filename2 = '/home/dlopezmo/datasetsV2/100box/xyrot_ball_in_square_10ballsv2.h5'

input_filename1 = '/home/dlopezmo/datasetsV2/100box/ball_in_polygon_3ballsv2_tbox100.h5'
output_filename1 = '/home/dlopezmo/datasetsV2/100box/xyrot_ball_in_polygon_3ballsv2.h5'

input_filename3 = '/home/dlopezmo/datasetsV2/100box/ball_in_ccpolygon_7ballsv2_tbox100.h5'
output_filename3 = '/home/dlopezmo/datasetsV2/100box/xyrot_ball_in_ccpolygon_7ballsv2.h5'

input_filename4 = '/home/dlopezmo/datasetsV2/100box/ball_in_polygonv2_tbox100.h5'
output_filename4 = '/home/dlopezmo/datasetsV2/100box/xyrot_ball_in_polygonv2.h5'



x_data1 = h5py.File(input_filename1, 'r')
x_data1 = x_data1['matrices']
y_data1 = h5py.File(output_filename1, 'r')
y_data1 = y_data1['matrices']
y_data1 = y_data1[:,:2]

x_data2 = h5py.File(input_filename2, 'r')
x_data2 = x_data2['matrices']
y_data2 = h5py.File(output_filename2, 'r')
y_data2 = y_data2['matrices']
y_data2 = y_data2[:,:2]

x_data3 = h5py.File(input_filename3, 'r')
x_data3 = x_data3['matrices']
y_data3 = h5py.File(output_filename3, 'r')
y_data3 = y_data3['matrices']
y_data3 = y_data3[:,:2]

x_data4 = h5py.File(input_filename4, 'r')
x_data4 = x_data4['matrices']
y_data4 = h5py.File(output_filename4, 'r')
y_data4 = y_data4['matrices']
y_data4 = y_data4[:,:2]
'''
x_data5 = h5py.File(input_filename5, 'r')
x_data5 = x_data5['matrices']
y_data5 = h5py.File(output_filename5, 'r')
y_data5 = y_data5['matrices']
y_data5 = y_data5[:,:2]

x_data6 = h5py.File(input_filename6, 'r')
x_data6 = x_data6['matrices']
y_data6 = h5py.File(output_filename6, 'r')
y_data6 = y_data6['matrices']
y_data6 = y_data6[:,:2]

x_data7 = h5py.File(input_filename7, 'r')
x_data7 = x_data7['matrices']
y_data7 = h5py.File(output_filename7, 'r')
y_data7 = y_data7['matrices']
y_data7 = y_data7[:,:2]

x_data8 = h5py.File(input_filename8, 'r')
x_data8 = x_data8['matrices']
y_data8 = h5py.File(output_filename8, 'r')
y_data8 = y_data8['matrices']
y_data8 = y_data8[:,:2]
'''
print('Filename1 Size')
print(x_data1.shape)
print(y_data1.shape)
print(x_data2.shape)
print(y_data2.shape)


#quit()
gc.collect()

# Subsample if need be, all datasets originally at 120 Hz subsamp_ratio = 1, set to 2 for 60Hz
subsamp_ratio = 3

x_data1 = x_data1[np.arange(0,x_data1.shape[0],subsamp_ratio),:,:,:]
y_data1 = y_data1[np.arange(0,y_data1.shape[0],subsamp_ratio),:]
x_data2 = x_data2[np.arange(0,x_data2.shape[0],subsamp_ratio),:,:,:]
y_data2 = y_data2[np.arange(0,y_data2.shape[0],subsamp_ratio),:]

x_data3 = x_data3[np.arange(0,x_data3.shape[0],subsamp_ratio),:,:,:]
y_data3 = y_data3[np.arange(0,y_data3.shape[0],subsamp_ratio),:]
x_data4 = x_data4[np.arange(0,x_data4.shape[0],subsamp_ratio),:,:,:]
y_data4 = y_data4[np.arange(0,y_data4.shape[0],subsamp_ratio),:]
print('Filename1 Size')
print(x_data1.shape)
print(x_data2.shape)
print(x_data3.shape)
print(x_data4.shape)
'''
x_data5 = x_data5[np.arange(0,x_data5.shape[0],subsamp_ratio),:,:,:]
y_data5 = y_data5[np.arange(0,y_data5.shape[0],subsamp_ratio),:]
x_data6 = x_data6[np.arange(0,x_data6.shape[0],subsamp_ratio),:,:,:]
y_data6 = y_data6[np.arange(0,y_data6.shape[0],subsamp_ratio),:]
x_data7 = x_data7[np.arange(0,x_data7.shape[0],subsamp_ratio),:,:,:]
y_data7 = y_data7[np.arange(0,y_data7.shape[0],subsamp_ratio),:]
x_data8 = x_data8[np.arange(0,x_data8.shape[0],subsamp_ratio),:,:,:]
y_data8 = y_data8[np.arange(0,y_data8.shape[0],subsamp_ratio),:]
'''

t_steps = 3
diff_step = 0
axis = 0

n_train = 700#2200
n_test = 120*6

### Reshape data
gc.collect()

# For t_step multi-step prediction
print('set1')
x_set1,y_set1 = fn.sequence_input(x_data1,y_data1,t_steps,axis,diff_step)
print('set2')
x_set2,y_set2 = fn.sequence_input(x_data2,y_data2,t_steps,axis,diff_step)

print('set3')
x_set3,y_set3 = fn.sequence_input(x_data3,y_data3,t_steps,axis,diff_step)
print('set4')
x_set4,y_set4 = fn.sequence_input(x_data4,y_data4,t_steps,axis,diff_step)
'''
print('set5')
x_set5,y_set5 = fn.sequence_input(x_data5,y_data5,t_steps,axis,diff_step)
print('set6')
x_set6,y_set6 = fn.sequence_input(x_data6,y_data6,t_steps,axis,diff_step)
print('set7')
x_set7,y_set7 = fn.sequence_input(x_data7,y_data7,t_steps,axis,diff_step)
print('set8')
x_set8,y_set8 = fn.sequence_input(x_data8,y_data8,t_steps,axis,diff_step)
'''

# For single output
pred_step = 0
y_set1 = y_set1[:,pred_step,:]
y_set2 = y_set2[:,pred_step,:]

y_set3 = y_set3[:,pred_step,:]
y_set4 = y_set4[:,pred_step,:]
'''
y_set5 = y_set5[:,pred_step,:]
y_set6 = y_set6[:,pred_step,:]
y_set7 = y_set7[:,pred_step,:]
y_set8 = y_set8[:,pred_step,:]
'''

print('Pre-norm Shapes:')
print(x_set1.shape)
print(x_set2.shape)
print(x_set3.shape)
print(x_set4.shape)
'''
print(x_set5.shape)
print(x_set6.shape)

print(y_set1.shape)
print(y_set2.shape)
print(y_set3.shape)
print(y_set4.shape)

print(y_set5.shape)
print(y_set6.shape)


quit()
'''
### Select Amount of Training Examples
nred = 0

x_train=x_set1[:n_train,:,nred:100-nred,nred:100-nred,:]
x_test=x_set1[n_train:,:,nred:100-nred,nred:100-nred,:]
y_train=y_set1[:n_train,:]
y_test=y_set1[n_train:,:]

x_set2=x_set2[:n_test,:,nred:100-nred,nred:100-nred,:]
y_set2=y_set2[:n_test,:]

x_set3=x_set3[:n_test,:,nred:100-nred,nred:100-nred,:]
y_set3=y_set3[:n_test,:]
x_set4=x_set4[:n_test,:,nred:100-nred,nred:100-nred,:]
y_set4=y_set4[:n_test,:]
print("train", x_train.shape)
x_train = np.concatenate((np.concatenate((np.concatenate((x_train,x_set2)),x_set3)),x_set4))
y_train = np.concatenate((np.concatenate((np.concatenate((y_train,y_set2)),y_set3)),y_set4))
print("train2", x_train.shape)
'''
x_set5=x_set5[:n_test,:,nred:100-nred,nred:100-nred,:]
y_set5=y_set5[:n_test,:]
x_set6=x_set6[:n_test,:,nred:100-nred,nred:100-nred,:]
y_set6=y_set6[:n_test,:]
x_set7=x_set7[:n_test,:,nred:100-nred,nred:100-nred,:]
y_set7=y_set7[:n_test,:]
x_set8=x_set8[:n_test,:,nred:100-nred,nred:100-nred,:]
y_set8=y_set8[:n_test,:]
'''

print('Pre-norm Shapes:')

print(x_set1.shape)
print(x_set2.shape)

print(x_set3.shape)
print(x_set4.shape)
'''
print(x_set5.shape)
print(x_set6.shape)
print(x_set7.shape)
print(x_set8.shape)

print(y_set1.shape)
print(y_set2.shape)

print(y_set3.shape)
print(y_set4.shape)

print(y_set5.shape)
print(y_set6.shape)
print(y_set7.shape)
print(y_set8.shape)
gc.collect()
'''

### Uncomment the following for single output pred
## also adjust net structure

#y_train =  y_train [:,1,:]

### Normalize inputs and outputs
xymax = 800
pixelmax = 255
norm_axis = (0,1,2,3,4)
xnorm_type = 'mean'
x_train,xdata_mean,xdata_range = fn.data_normalize_v2(x_train,norm_type= xnorm_type,data_mean='non',data_range='non',norm_axis=norm_axis)
y_train,ydata_mean,ydata_range = fn.data_normalize(y_train,norm_type='max',data_mean=0,data_range=xymax)
gc.collect()
x_test,xdata_mean,xdata_range = fn.data_normalize_v2(x_test,norm_type= xnorm_type ,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_test,ydata_mean,ydata_range = fn.data_normalize(y_test,norm_type='max',data_mean=0,data_range=xymax)
gc.collect()
x_set2,xdata_mean,xdata_range = fn.data_normalize_v2(x_set2,norm_type= xnorm_type,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_set2,ydata_mean,ydata_range = fn.data_normalize(y_set2,norm_type='max',data_mean=0,data_range=xymax)
gc.collect()
x_set3,xdata_mean,xdata_range = fn.data_normalize_v2(x_set3,norm_type= xnorm_type ,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_set3,ydata_mean,ydata_range = fn.data_normalize(y_set3,norm_type='max',data_mean=0,data_range=xymax)
x_set4,xdata_mean,xdata_range = fn.data_normalize_v2(x_set4,norm_type= xnorm_type,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_set4,ydata_mean,ydata_range = fn.data_normalize(y_set4,norm_type='max',data_mean=0,data_range=xymax)
'''
x_set5,xdata_mean,xdata_range = fn.data_normalize_v2(x_set5,norm_type= xnorm_type,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_set5,ydata_mean,ydata_range = fn.data_normalize(y_set5,norm_type='max',data_mean=0,data_range=xymax)
x_set6,xdata_mean,xdata_range = fn.data_normalize_v2(x_set6,norm_type= xnorm_type,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_set6,ydata_mean,ydata_range = fn.data_normalize(y_set6,norm_type='max',data_mean=0,data_range=xymax)
x_set7,xdata_mean,xdata_range = fn.data_normalize_v2(x_set7,norm_type= xnorm_type,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_set7,ydata_mean,ydata_range = fn.data_normalize(y_set7,norm_type='max',data_mean=0,data_range=xymax)
x_set8,xdata_mean,xdata_range = fn.data_normalize_v2(x_set8,norm_type= xnorm_type,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)
y_set8,ydata_mean,ydata_range = fn.data_normalize(y_set8,norm_type='max',data_mean=0,data_range=xymax)

# Set for single output


print('Post-norm Shapes:')
print(x_set5.shape)
print(x_test.shape)
print(x_train.shape)
print(y_set5.shape)

print(y_test.shape)
print(y_train.shape)
#print(y_data[900:910,:])
'''
gc.collect()

##### CNN-LSTM Parameter Control Module #####
n_unitsFC1 = 100
n_unitsFC2 = 100
LSTM_neurons = 100
n_convpool_layers = 1
n_convlayers = 3
n_reglayers = 1
max_poolsize = (3,3)
train_set_ratio = 0.7
valid_set_ratio = 0.15
drop_rate = 0.0
n_filters = 100
kernel_size = (3,3)
input_strides = 1
kernel_init = 'glorot_uniform'
cost_function = 'mean_squared_error'
batch_size = 10
n_epochs = 30
n_output = 2
n_post_test = 5000
early_stop_delta = 0.000001 # 0.0025 change or above is considered improvement
early_stop_patience = 10 # keep optimizing for 10 iterations under "no improvement"
out_filepath = 'trained_nets/file_003_5tsteps_120Hz_'+str(pbs_jobid)+'.h5'
outdata_filename = 'output_data/train3b_10bsq_60Hz_sinout_smrnorm_3steps_15nred'

print('Network Params:')
print 'LSTM_neurons',LSTM_neurons
print 'cost_function:', cost_function
print 'n_unitsFC1:',n_unitsFC1
print 'n_unitsFC2:',n_unitsFC2
print 'LSTM_neurons:', LSTM_neurons
print 'batch_size:', batch_size
print 'kernel_size:', kernel_size
print 'n_filters:', n_filters
print 'max_poolsize:', max_poolsize
print 'drop_rate:', drop_rate
print 'early_stop_delta:', early_stop_delta
print 'early_stop_patience:', early_stop_patience


# We create a layer which take as input movies of shape
# (n_frames, width, height, channels(RxGxBxXxY)) and returns the X,Y
# coordinates of moving object

model = Sequential()

# define CNN model architecture


model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                  kernel_initializer= kernel_init,
                  activation='relu'),input_shape=(t_steps,100-nred*2,100-nred*2,5)))
model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))



model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                  kernel_initializer= kernel_init,
                  activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))



model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                  kernel_initializer= kernel_init,
                  activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))


model.add(TimeDistributed(Flatten()))

model.add(TimeDistributed(Dense(n_unitsFC1, activation='relu',kernel_initializer= kernel_init)))
model.add(TimeDistributed(Dense(n_unitsFC1, activation='relu',kernel_initializer= kernel_init)))
model.add(CuDNNLSTM(LSTM_neurons, return_sequences=False,kernel_initializer= kernel_init))
model.add(Dense(n_output, activation='linear',kernel_initializer= kernel_init))

model.summary()

def MAE(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true)/K.abs(y_true))

model.compile(loss=cost_function, optimizer='adam', metrics=[MAE])

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=early_stop_delta, patience=early_stop_patience, verbose=1, mode='auto')

# Train the network

history = model.fit(x_train, y_train, epochs=n_epochs,
        batch_size=batch_size, verbose=2,
        callbacks=[earlyStopping],
        validation_split = valid_set_ratio)
#        validation_data =[inp_valid,out_valid])

# Save model
#model.save(out_filepath)

gc.collect()

## Post-Processing
print('Test Case 1:')
print(input_filename1)
print('Actual Values')
print(y_test[110:112,:])
print('Predicted Values')
print((model.predict_on_batch(x_test[110:112,:,:,:,:])))

score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

## Post-Processing
print('Test Case 2:')
print(input_filename2)
print('Actual Values')
print(y_set2[110:112,:])
print('Predicted Values')
print((model.predict_on_batch(x_set2[110:112,:,:,:,:])))

score = model.evaluate(x_set2, y_set2, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

gc.collect()

## Post-Processing
print('Test Case 3:')
print(input_filename3)
print('Actual Values')
print(y_set3[10:12,:])
print('Predicted Values')
print((model.predict_on_batch(x_set3[10:12,:,:,:,:])))

score = model.evaluate(x_set3, y_set3, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

## Post-Processing
print('Test Case 4:')
print(input_filename4)
print('Actual Values')
print(y_set4[10:12,:])
print('Predicted Values')
print((model.predict_on_batch(x_set4[10:12,:,:,:,:])))

score = model.evaluate(x_set4, y_set4, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

gc.collect()
'''
## Post-Processing
print('Test Case 5:')
print(input_filename5)
print('Actual Values')
print(y_set5[10:12,:])
print('Predicted Values')
print((model.predict_on_batch(x_set5[10:12,:,:,:,:])))

score = model.evaluate(x_set5, y_set5, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]


print('Test Case 6:')
print(input_filename6)
print('Actual Values')
print(y_set6[10:12,:])
print('Predicted Values')
print((model.predict_on_batch(x_set6[10:12,:,:,:,:])))

score = model.evaluate(x_set6, y_set6, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

print('Test Case 7:')
print(input_filename7)
print('Actual Values')
print(y_set7[10:12,:])
print('Predicted Values')
print((model.predict_on_batch(x_set7[10:12,:,:,:,:])))

gc.collect()

score = model.evaluate(x_set7, y_set7, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

print('Test Case 8:')
print(input_filename8)
print('Actual Values')
print(y_set8[10:12,:])
print('Predicted Values')
print((model.predict_on_batch(x_set8[10:12,:,:,:,:])))

score = model.evaluate(x_set8, y_set8, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

print('Saving data..')

gc.collect()

#env_jobid = os.environ['PBS_JOBID']
#pbs_jobid = [int(s) for s in env_jobid.split('.') if s.isdigit()]

#target_test = y_set3
#output_test = model.predict_on_batch(x_set3)
fileformat = '.h5'
#print(input_filename3)

#hf = h5py.File(outdata_filename + str(pbs_jobid) + fileformat, 'w')
#hf.create_dataset('target_set3', data=target_test)
#hf.create_dataset('output_set3', data=output_test)

target_test = y_set2
output_test = model.predict(x_set2)
print(input_filename2)
#filename = 'output_data/train3ball_ballbounce_singleout'
fileformat = '.h5'

hf = h5py.File(outdata_filename + str(pbs_jobid) + fileformat, 'w')
hf.create_dataset('target_set2', data=target_test)
hf.create_dataset('output_set2', data=output_test)

target_test = y_test[0:n_test,:]
output_test = model.predict(x_test[0:n_test,:,:,:,:])
print(input_filename1)
print(target_test.shape)
print(output_test.shape)
hf.create_dataset('target_test', data=target_test)
hf.create_dataset('output_test', data=output_test)

#with open(filename+fileformat, 'wb') as csvfile:
#        filewriter = csv.writer(csvfile)
#        for j in range(0,out.shape[0]):
#            filewriter.writerow(out[j,:])



print('Network Params:')

print 'cost_function:', cost_function
print 'n_unitsFC1:',n_unitsFC1
print 'n_unitsFC2:',n_unitsFC2
print 'LSTM_neurons:', LSTM_neurons
print 'batch_size:', batch_size
print 'kernel_size:', kernel_size
print 'n_filters:', n_filters
print 'max_poolsize:', max_poolsize
print 'drop_rate:', drop_rate
print 'early_stop_delta:', early_stop_delta
print 'early_stop_patience:', early_stop_patience


# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions

#for j in range(16):
#    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
#    new = new_pos[::, -1, ::, ::, ::]
#    track = np.concatenate((track, new), axis=0)

'''
