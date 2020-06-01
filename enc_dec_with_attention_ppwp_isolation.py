# Commented out IPython magic to ensure Python compatibility.
#everything we will import to use later
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import model_from_json
# %matplotlib inline

import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
#check our version
print(tf.VERSION)
print(tf.keras.__version__)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def rescale(feature_data,test_targets,predictions, scaling_object,index):
    '''Flattens and rescales test and prediction data back to the original scale.
    Given that the test data and predictions do not have the same shape as the original feature data, we need 
    to "pad" these two datasets with the original column numbers of the feature data, 
    as well as have the test and prediction data occupy the same positions of their respective 
    target data columns so the rescale is done properly. 
    The below code includes one way to correctly do this padding.
    
    INPUTS: training or test feature data (it doesn't matter--we just need the same number of columns)
    test targets, and predictions, all in 3D tensor form. Also, the scaling object used 
    for the original transformation'''
    
    #flatten predictions and test data
    predict_flat = predictions.reshape(predictions.shape[0]*predictions.shape[1],predictions.shape[2])
    y_test_flat = test_targets.reshape(test_targets.shape[0]*test_targets.shape[1],test_targets.shape[2])

    #flatten the features dataframe. This has the dimensions we want.
    flattened_features = pd.DataFrame(feature_data.reshape(feature_data.shape[0]*feature_data.shape[1],
                                                           feature_data.shape[2]))
    
    #if we want to predict long sequences these will likely be longer than the length of the features.
    #we just add some zeros here to pad out the feature length to match. We'll then convert these to NaN's
    #so they're ignored. Again, we just want to inherient the 7 col 
    #structure of the feature data--not the values. This is the most foolproof method
    #I've found through trial and error.
    
    if len(flattened_features) < len(y_test_flat):
        print('Length of targets exceeds length of features. Now padding...\n')
        extra_rows = pd.DataFrame(np.zeros((len(y_test_flat),flattened_features.shape[1])))
        flattened_features = pd.concat([flattened_features,extra_rows],axis=0)
        flattened_features[flattened_features==0]=np.nan
        
    #make a start column, this is the index where we begin to repopulate the target cols with the 
    #data we want to rescale
    start_col = feature_data.shape[2]-test_targets.shape[2]
    total_col = feature_data.shape[2]
    
    #make trimmed feature copies of equal length as the test data and predictions lengths, 
    #and leave out the original target data... we will replace these cols with the test and prediction data
    flattened_features_test_copy = flattened_features.iloc[:len(y_test_flat),:start_col]
    flattened_features_pred_copy = flattened_features.iloc[:len(y_test_flat),:start_col]
    #print((flattened_features_pred_copy.values))
    
    for i in range(start_col,total_col):
        #reassign targets cols
        flattened_features_test_copy[i] = y_test_flat[:,i-start_col] 
        flattened_features_pred_copy[i] = predict_flat[:,i-start_col]
        #by specifying 'i - start col', we are making sure the target column being 
        #repopulated is the matching target taken from the test data or predictions.
        #Ex: if the start col is 4, then we want to assign the first column of the test and pred data--
        #this is index 0, and 4-4 = 0.
        
    #We now have the correct dimensions, so we can FINALLY rescale
    y_test_rescale = scaling_object.inverse_transform(flattened_features_test_copy)
    preds_rescale = scaling_object.inverse_transform(flattened_features_pred_copy)
    
    #just grab the target cols.
    y_test_rescale = y_test_rescale[:,start_col:]
    preds_rescale = preds_rescale[:,start_col:]
    
    preds_rescale = pd.DataFrame(preds_rescale,index=index)
    y_test_rescale = pd.DataFrame(y_test_rescale,index=index)
    
    
     #before we return the dataframes, check and see if predictions or test data have null values.
    if preds_rescale.isnull().values.any()==True:
        print('Keras predictions have NaN values present. Deleting...')
        print('Current shape: ' + str(preds_rescale.shape))
        nans = np.argwhere(np.isnan(preds_rescale.values)) #find nulls
        #delete make sure values are deleted from both
        y_test_rescale = np.delete(y_test_rescale.values, nans,axis=0)
        preds_rescale = np.delete(preds_rescale.values, nans, axis=0)
        index = np.delete(index, nans, axis=0)
        #turn back into dataframe in case next condition is also true
        preds_rescale = pd.DataFrame(preds_rescale, index=index) 
        y_test_rescale = pd.DataFrame(y_test_rescale, index=index)
        print('New Shape: ' + str(preds_rescale.shape))
        
    
    if y_test_rescale.isnull().values.any()==True:
        print('Test data still have NaN values present. Deleting...')
        print('Current shape: ' + str(y_test_rescale.shape))
        nans = np.argwhere(np.isnan(y_test_rescale.values)) 
        #same as above
        y_test_rescale = np.delete(y_test_rescale.values, nans,axis=0)
        preds_rescale = np.delete(preds_rescale.values, nans, axis=0)
        index = np.delete(index, nans, axis=0)
        # make into DataFrame this time to guarantee the below return statement won't spit an error
        y_test_rescale = pd.DataFrame(y_test_rescale,index=index)
        preds_rescale = pd.DataFrame(preds_rescale,index=index)
        print('New shape: ' + str(y_test_rescale.shape))
    
    print('test data new shape: ' + str(y_test_rescale.shape))
    print('prediction new shape: ' + str(preds_rescale.shape))
    return y_test_rescale,preds_rescale

def lstm_prep(data_index,data,ntargets,ninputs,noutputs=1,show_progress=False):
    '''Prepares and reshapes data for use with an LSTM. Outputs features, targets,
    and the original data indices of your target values for visualization later. Requires that 
    the targets are the last N columns in your dataset.
    
    NOTE: The applies a moving window approach at intervals of the output steps, such that 
    you group the previous timesteps of inputs for your features (whatever length you choose),set the next 
    X timesteps of target values as outputs (again, whatever you want), and then move the window X (noutputs)
    timesteps in the future to repeat the process. Analogous to a cnn kernal with a stride equal to the output length. 
    I wrote this to automate and quickly change between varying input and output sequence lengths, 
    but wanted to avoid overlapping values typical in a moving window approach. 
    Having these non-overlapping values just makes plotting easier. 
    So far I have yet to see a need for more samples, which I understand is why the 
    moving window approach is typicallyimplemented.'''
    
    target_data = data[:,-ntargets:]
    features = np.empty((ninputs,data.shape[1]), int)
    targets = np.empty((noutputs,ntargets),int)
    for i in range(ninputs,(len(data)-noutputs),noutputs): 
        if show_progress==True:
            print('current index: '+str(i))
            print('feature indices: '+str(i-ninputs)+' to '+str(i-1))
            print('target indices: '+str(i)+' to '+str(i+noutputs-1))
        #print(i)
        temp_feature_matrix = data[(i-ninputs):i]
        temp_target_matrix = target_data[(i):(i+noutputs)]
        #print(i+noutputs)
        features = np.vstack((features, temp_feature_matrix))
        targets = np.vstack((targets,temp_target_matrix))
    
    last_index = i+noutputs
    features = features.reshape((int(features.shape[0]/ninputs),ninputs,features.shape[1]))
    targets = targets.reshape(int(targets.shape[0]/noutputs),noutputs,targets.shape[1])
    
    target_indices = data_index[ninputs:last_index]
    
    return features[1:], targets[1:], target_indices

def rectify_cnn_data(predictions, targets, num_targets, noutputs):
    preds = np.empty((noutputs,num_targets), int)
    tests = np.empty((noutputs,num_targets), int)
    for row in range(0,predictions.shape[0]):
        pred_t = np.transpose(np.array(np.split(predictions[row],num_targets)))
        #print('preds: '+ str(pred_t))
        preds = np.vstack((preds, pred_t))

        test_t = np.transpose(np.array(np.split(targets[row],num_targets)))
        #print('tests: '+ str(test_t))
        tests = np.vstack((tests, test_t))
    #print(preds)
    return preds[noutputs:], tests[noutputs:]


def mse_nan(y_true, y_pred):
    masked_true = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true)
    masked_pred = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_pred)
    return K.mean(K.square(masked_pred - masked_true), axis=-1)

def mae_nan(y_true, y_pred):
    masked_true = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true)
    masked_pred = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_pred)
    return K.mean(abs(masked_pred - masked_true), axis=-1)


def evaluate_forecasts(actual, predicted): #from Jason Brownlee's blog
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        #print('mse: '+ str(mse))
        # calculate rmse
        rmse = np.sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    
    print(scores)
    return score, scores

def threshold_rmse_eval(tests, preds, threshold):
    pred_pos_all = []
    y_pos_all = []
    rmse = []
    for i in range(preds.shape[1]):
        #grab individual cols
        data_slice_test = tests[:,i]
        data_slice_pred = preds[:,i]
        
        if max(data_slice_test) < threshold:
          print("Threshold too high for column: "+str(i))
          rmse.append('N/A')
          y_pos_all.append('N/A')
          pred_pos_all.append('N/A')
          continue
        
        #This avoids a warning for the np.where query a couple of lines down...
        test_nans = np.where(np.isnan(data_slice_test)) #find nans and replace with dummy value. 
        tests[test_nans] = threshold*100
        
        #find all values greater than or equal to your threshold value
        pos_test = np.where(data_slice_test >= threshold)
        y_pos = data_slice_test[pos_test] #get those values from the test data
        pred_pos = data_slice_pred[pos_test] #get the equivalent values from the predictions

        #calculate mse, rmse
        mse = mean_squared_error(y_pos, pred_pos)
        rmse_val = np.sqrt(mse)
        
        #append all values to our respective lists
        rmse.append(rmse_val)
        y_pos_all.append(y_pos)
        pred_pos_all.append(pred_pos)
    
    print('per target rmse: ' + str(rmse))
    return np.asarray(rmse),np.asarray(y_pos_all),np.asarray(pred_pos_all)
    
    #make a list of rmse per datapoint
    diff_array = np.empty((1,tests.shape[1]),int)
    for row in range(tests.shape[0]):
        diffs = []
        for col in range(tests.shape[1]):
            #print(tests[row,col])
            #print(preds[row,col])
            diff = tests[row,col] - preds[row, col]
            diffs.append(diff)
        diffs = np.array(diffs)
        diff_array = np.vstack((diff_array,diffs))
    
    return y_pos_all, pred_pos_all, rmse, diff_array[1:]

def naive_forecast(test_data,backsteps,forward_steps):
    back_array = np.arange(0,backsteps+1)
    naive_preds = np.zeros(shape=(test_data.shape[0],test_data.shape[1]))
    for row in range(0,test_data.shape[0],forward_steps):
        #print(row)
        for col in range(test_data.shape[1]):
            if row in back_array:
                naive_preds[row,col] = test_data[row,col]
            else:
                naive_preds[row,col] = np.mean(test_data[(row-backsteps):row,col])
                for i in range(1,forward_steps):
                    row_index = row+i
                    if row_index == len(test_data):
                        break
                    else:
                        naive_preds[(row_index),col] = np.mean(naive_preds[(row_index-backsteps):row_index,col])

    return naive_preds

def add_max_rainfall(data,interval,rain_col,noise=False):
    #print(len(data))
    '''takes times series data as input, and calculates the maximum and total rainfall values 
    for a fixed interval, based on the column index of your rainfall (rain_col).

    This function assumes you are using pandas, but can be modified for numpy arrays, which will perform faster
    for larger datasets'''
    rain_total = np.array(np.zeros(len(data)))
    rain_max = np.array(np.zeros(len(data)))
    
    if noise==True:
      print('adding with noise')
      for row in range(0,len(data),1):
        print('no test')
        if row >= (len(data) - interval):
            rain_total[row] = (np.sum(data.iloc[row:len(data),rain_col]))*np.random.randint(0.75,1.25)
            rain_max[row] = (max(data.iloc[row:len(data),rain_col]))*np.random.randint(0.75,1.25)
            
        rain_total[row] = np.sum(data.iloc[row:row+interval,rain_col])*np.random.randint(0.75,1.25)
        rain_max[row] = max(data.iloc[row:row+interval,rain_col])*np.random.randint(0.75,1.25)



    else:
      #print('adding with no noise')
      for row in range(0,len(data),1):
          
          if row >= (len(data) - interval):
              rain_total[row] = np.sum(data.iloc[row:len(data),rain_col])
              rain_max[row] = np.nanmax(data.iloc[row:len(data),rain_col])
          
          rain_total[row] = np.sum(data.iloc[row:row+interval,rain_col])
          rain_max[row] = np.nanmax(data.iloc[row:row+interval,rain_col])
          #print(np.nanmax(data.iloc[row:row+interval,rain_col]))

    data[str(interval)+'hr_total_rfall'] = rain_total
    data[str(interval)+'hr_max_rfall'] = rain_max
    
    return data

from google.colab import files
uploaded = files.upload()

ocr = pd.read_csv('SP1.txt',index_col='timestamp_utc',sep='\t')
datetime = pd.to_datetime(ocr.index,dayfirst=False)
ocr = ocr.set_index(datetime)
mask_cols = ['porePressure_63cm_kPa','porePressure_113cm_kPa','porePressure_244cm_kPa']
ocr[mask_cols] = ocr[mask_cols].mask(ocr[mask_cols] < -14)

# set dates also as feature cols
ocr['date'] = ocr.index.values.astype(float)

# add in forecasted rainfall
# this will take a while
for i in range(2,38,2):
  ocr = add_max_rainfall(ocr,i,0,noise=False)

# look at all cols
ocr.columns

# rearrange so we have all the feature cols as we want them
ocr = ocr.loc[:,['date', 'RG_mm', '2hr_total_rfall', '2hr_max_rfall', '4hr_total_rfall', '4hr_max_rfall',
       '6hr_total_rfall', '6hr_max_rfall', '8hr_total_rfall', '8hr_max_rfall',
       '10hr_total_rfall', '10hr_max_rfall', '12hr_total_rfall',
       '12hr_max_rfall', '14hr_total_rfall', '14hr_max_rfall',
       '16hr_total_rfall', '16hr_max_rfall', '18hr_total_rfall',
       '18hr_max_rfall', '20hr_total_rfall', '20hr_max_rfall',
       '22hr_total_rfall', '22hr_max_rfall', '24hr_total_rfall',
       '24hr_max_rfall', '26hr_total_rfall', '26hr_max_rfall',
       '28hr_total_rfall', '28hr_max_rfall', '30hr_total_rfall',
       '30hr_max_rfall', '32hr_total_rfall', '32hr_max_rfall',
       '34hr_total_rfall', '34hr_max_rfall', '36hr_total_rfall',
       '36hr_max_rfall', 'VWC_20cm', 'VWC_50cm', 'VWC_93cm',
       'porePressure_63cm_kPa', 'porePressure_113cm_kPa',
       'porePressure_244cm_kPa']]

# create a feature scaling object for use later
data = ocr
data_scaling = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(data.iloc[:,:])
data_scaled = pd.DataFrame(data_scaling.transform(data.iloc[:,:]),index=ocr.index)

# now create our features and targets
all_features, all_targets, all_target_indices = lstm_prep(
    ocr.index.values,ocr.values,3,36,36)

# set prediction intervals for each batch... this is useful for plotting later
intervals = np.zeros(len(all_target_indices))
for i in range(0,len(intervals),all_targets.shape[1]):
    intervals[i]=1

intervals[intervals==0] = np.nan

binary_indices = np.copy(all_target_indices)

for i in range(0,len(binary_indices),all_targets.shape[1]):
    binary_indices[i]=np.datetime64("NaT")

def isolate_threshold_rainfall(feature_data,target_data,target_indices,rain_col,
                               threshold_value):
  
  '''Sorts feature and target data based on 
  a user-specificed threshold. It returns 4 lists with 
  the above and below threshold values, respectively, in order:
  feature, target, target indices (features have the index built in),
  a list of the max rainfall from each positive batch, and the same for the 
  negative batches. The last two are just helpful for making the algorithm
  correctly sorted based on a fixed threshold, i.e., the negatives only have
  max values below the threshold.
  
  Input "target_columns" specifies the number of target features present,
  and assumes these are placed as the last members of your feature data'''


  # reshape 1D target indices to be in 'batch' form similar to features/targets
  # every new row now corresponds to the n timesteps of each batch
  target_indices_re = np.reshape(target_indices, (target_data.shape[0],
                                                  target_data.shape[1]))
  thresh_features = []
  thresh_targets = []
  arrays_to_delete = []
  ppwp_indices = []
  pthresh_list = []
  nthresh_list = []

  for batch in range(0,target_data.shape[0],1):
    print('batch: '+str(batch))
    target_array = target_data[batch,:,:]
    
    # create 'feature array' that just has the col of interest
    feature_array = feature_data[batch,:,rain_col]

    # get all values in the current array over threshold, if any
    values_above_thresh = (feature_array[feature_array >= threshold_value])

    if len(values_above_thresh) != 0:
      print('values above thresh: '+ str(len(values_above_thresh)))
      print('max: '+str(np.nanmax(feature_array)))
      print('min: ' +str(np.nanmin(feature_array)))
      pthresh_list.append(np.nanmax(feature_array))
      #print(np.nanmax(feature_array))
      #print(len(values_above_thresh))
      # if we have positives, add that 'batch' to our new ppwp datasets
      thresh_features.append(feature_data[batch,:,:])
      thresh_targets.append(target_data[batch,:,:])
      arrays_to_delete.append(batch)
      ppwp_indices.append(target_indices_re[batch,:])
      #print("values appended")

    else:
      print('none above threshold')
      #print('values above thresh: '+ str(len(values_above_thresh)))
      nthresh_list.append(np.nanmax(feature_array))
      #print('max: ' +str(np.nanmax(feature_array)))
      #print('no threshold met')
      continue
  
  # deleting indices no longer in feature/target data
  npwp_indices = np.delete(target_indices_re,np.asarray(arrays_to_delete),0)

  #reshaping to correct form for use later
  npwp_indices_2d = npwp_indices.reshape(
      npwp_indices.shape[0]*npwp_indices.shape[1])
  
  ppwp_indices = (np.asarray(ppwp_indices))

  #same reshaping process
  ppwp_indices_2d = ppwp_indices.reshape(
      ppwp_indices.shape[0]*ppwp_indices.shape[1])
  
  # delete positive pressures from our dataset
  feature_data_no_thresh = np.delete(feature_data, 
                                     np.asarray(arrays_to_delete),0)
  target_data_no_thresh = np.delete(target_data, 
                                    np.asarray(arrays_to_delete),0)

  # group items in list for simplicity
  ppwp_items = [np.asarray(thresh_features), np.asarray(thresh_targets), 
          ppwp_indices_2d]
  
  npwp_items = [feature_data_no_thresh, target_data_no_thresh,
                     npwp_indices_2d]
                     

  print('pthresh max: '+str(np.nanmax(pthresh_list)))
  print('nthresh max: '+str(np.nanmax(nthresh_list)))
  return ppwp_items, npwp_items, pthresh_list, nthresh_list

# create separate feature and target datasets, based on which target
# sequences feature positive pore-water pressures
ppwp_items, npwp_items, pthresh, nthresh = isolate_threshold_rainfall(all_features,all_targets,
                                                    all_target_indices,-7,4)

#plot histogram of max below 4mm threshold 36hr forecast... 
# correctly, there is nothing above the 4mm I specified
plt.hist(nthresh)

ppwp_f, ppwp_t, ppwp_ti = ppwp_items
npwp_f, npwp_t, npwp_ti = npwp_items

# now we temporarily reshape features to 2D arrays, just to rescale
ppwp_f_re = np.reshape(ppwp_f,(ppwp_f.shape[0]*ppwp_f.shape[1],ppwp_f.shape[2]))
ppwp_f_index = pd.to_datetime(ppwp_f_re[:,0]) # track feature index for convenience

#scale
ppwp_f_scaled = pd.DataFrame(data_scaling.transform(ppwp_f_re),index=ppwp_f_index)
ppwp_f_scaled_fna = ppwp_f_scaled.fillna(-1)


# same thing as above for negative pressures
npwp_f_re = np.reshape(npwp_f,(npwp_f.shape[0]*npwp_f.shape[1],npwp_f.shape[2]))
npwp_f_index = pd.to_datetime(npwp_f_re[:,0]) # feature index for below threshold
npwp_f_scaled = pd.DataFrame(data_scaling.transform(npwp_f_re),index=npwp_f_index)
npwp_f_scaled_fna = npwp_f_scaled.fillna(-1)

# shape back into 3D
# yes, this is repetitive
ppwp_f_scaled_re = np.reshape(ppwp_f_scaled_fna.values,
                              (ppwp_f.shape[0],ppwp_f.shape[1],ppwp_f.shape[2]))

npwp_f_scaled_re = np.reshape(npwp_f_scaled_fna.values,
                              (npwp_f.shape[0],npwp_f.shape[1],npwp_f.shape[2]))

# as final check... let's look at the 36hr_max_rfall 
# upon which we base our thresholds for all the feature data
# notice our trianing data has all forecasts below 4mm.
print(np.nanmax(ocr.loc[npwp_f_index,'36hr_max_rfall']))
plt.scatter(npwp_f_index,(ocr.loc[npwp_f_index,'36hr_max_rfall']))

# this is a bit of an arbitrary split
# mostly lets just see how well the model performs on below threshold data
split = 0.7
train_features = npwp_f_scaled_re[:int((len(npwp_f_scaled_re)*split))]
test_features = npwp_f_scaled_re[int((len(npwp_f_scaled_re)*split)):]

train_targets = npwp_t[:int((len(npwp_f_scaled_re)*split))]
test_targets = npwp_t[int((len(npwp_f_scaled_re)*split)):]
test_indices = npwp_ti[-test_targets.shape[0]*test_targets.shape[1]:]

"""Create our Encoder:"""

# Define an input shape. This is the tensor shape our model expects from now on.
# This is essential for a stateful model
batch = None
n_units = 36

encoder_inputs = Input(batch_input_shape=(batch,train_features.shape[1],train_features.shape[2]))

encoder_lstm = LSTM(n_units, return_state=True,stateful=False,return_sequences=True) # define encoder

# connect encoding layer to our inputs, return all states
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs) 

encoder_states = [state_h, state_c]

encoder_lstm = Bidirectional(LSTM(units=n_units, return_sequences=True, return_state=True)) # Bidirectional
encoder_outputs, fstate_h, fstate_c, bstate_h, bstate_c = encoder_lstm((encoder_inputs))
state_h = Concatenate()([fstate_h,bstate_h])
state_c = Concatenate()([bstate_h,bstate_c])

encoder_states = [state_h, state_c]

"""Create Decoder:"""

# Define inputs to the decoder.
decoder_inputs = Input(batch_input_shape=(batch,None,train_targets.shape[2]))
#decoder_inputs = Input(batch_shape=(batch,features.shape[1],features.shape[2]))

#decoder_inputs = Input(batch_shape=(batch,None,train_targets.shape[2]))
#dec_input = np.zeros((train_features.shape[0],train_targets.shape[1],train_features.shape[2]))


# Create Decoder...
decoder_lstm = (LSTM(n_units, return_state=True, return_sequences=True,stateful=False))
# Important step: connect Decoder to our input layers and use the hidden and cell states 
# from the encoder to instantiate this layer

decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

#create attention layer
# ----------------------
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder_outputs], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_outputs])
# ----------------------

decoder_dense1 = (Dense(150, activation='tanh'))
dropout = Dropout(0.5)
decoder_dense2 = Dense(train_targets.shape[2],activation='linear')

dense1 = decoder_dense1(decoder_combined_context)
drop = dropout(dense1)

decoder_outputs = decoder_dense2(drop)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.compile(optimizer='adam',
             loss=mse_nan,
             metrics=[mse_nan])

model.summary()

#define inference ('inf') model:
encoder_model = Model(encoder_inputs, [encoder_outputs,encoder_states]) 
# ^^^ set up a separate encoding model. This just outputs our encoder states

inf_encoder_outputs, inf_encoder_states = encoder_model(encoder_inputs)

decoder_state_input_h = Input(shape=(n_units,)) # define shape of hidden state
decoder_state_input_c = Input(shape=(n_units,)) # same thing here for cell state
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# ^^^ set as state shapes, which tells our decoder to accepts inputs states of the specificed size

# create our decoding layer. accepts same shape as decoder inputs and encoder states
inf_decoder_outputs, inf_state_h, inf_state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# save decoder output states. We'll use these as the input states for our decoder for predicting each next timestep
# after the initial input of our encoder states
decoder_states = [inf_state_h, inf_state_c]

inf_attention = dot([inf_decoder_outputs, inf_encoder_outputs], axes=[2, 2])
inf_attention = Activation('softmax')(inf_attention)

inf_context = dot([inf_attention, inf_encoder_outputs], axes=[2,1])
inf_decoder_combined_context = concatenate([inf_context, inf_decoder_outputs])


inf_dense1 = decoder_dense1(inf_decoder_combined_context)
inf_drop = dropout(inf_dense1)

inf_final_outputs = decoder_dense2(inf_drop)
# finally, instantiate our decoder model. Inputs are the original sequence + the encoder states. 
# outputs: sequence prediction + the states used for the decoder
decoder_model = Model([encoder_inputs,decoder_inputs] + decoder_states_inputs, [inf_final_outputs]+decoder_states)

decoder_model.summary()

# Let's define a small function that predicts based on the trained encoder and decoder models
# Function adapted from https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction
def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict, 
            num_features_to_predict, batch_size=None):
    """Predict time series with encoder-decoder.
    
    Uses the encoder and decoder models previously trained to predict the next
    num_steps_to_predict values of the time series.
    
    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder model.
    decoder_predict_model: The Keras decoder model.
    num_steps_to_predict: The number of steps in the future to predict
    num_features_to_predict: The number of features we want to predict per timestep
    
    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    y_predicted = []

    # Encode the values as a state vector
    enc_outputs_and_states = encoder_predict_model.predict(x,batch_size=batch)
    #print(len(enc_outputs_and_states))

    # The states must be a list
    if not isinstance(enc_outputs_and_states, list):
        enc_outputs_and_states = [enc_outputs_and_states]

    enc_outputs = enc_outputs_and_states[0]
    states = enc_outputs_and_states[1:]
    #decoder_input = np.zeros((x.shape[0], 1, num_features_to_predict))
    decoder_input = x[:,-1:,-num_features_to_predict:] 
    # '-num_features_to_predict:' assumes the targets are the final columns

    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
        [x, decoder_input] + states, batch_size=batch)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)
        #decoder_input = output

    return np.concatenate(y_predicted, axis=1)

for i in range(1000):
    print('epoch: '+str(i))
    model.fit([train_features,train_features[:,-36:,-3:]], train_targets, 
              epochs=1, batch_size=100, verbose=1, shuffle=True)
    #model.reset_states()

"""Evaluate Model"""

npwp_predictions = predict(test_features, encoder_model, decoder_model, 
                      num_steps_to_predict=train_targets.shape[1], num_features_to_predict=train_targets.shape[2],
                     batch_size=batch)

ppwp_predictions = predict(ppwp_f_scaled_re, encoder_model, decoder_model, 
                      num_steps_to_predict=train_targets.shape[1], num_features_to_predict=train_targets.shape[2],
                     batch_size=batch)

ppwp_preds = pd.DataFrame(
    np.reshape(ppwp_predictions,(ppwp_predictions.shape[0]*ppwp_predictions.shape[1],
                                 ppwp_predictions.shape[2])),index=ppwp_ti)
ppwp_tests = pd.DataFrame(np.reshape(ppwp_t,(ppwp_t.shape[0]*ppwp_t.shape[1],ppwp_t.shape[2])),index=ppwp_ti)

npwp_preds = pd.DataFrame(
    np.reshape(npwp_predictions,(npwp_predictions.shape[0]*npwp_predictions.shape[1],
                                 npwp_predictions.shape[2])),index=test_indices)
npwp_tests = pd.DataFrame(np.reshape(test_targets,(test_targets.shape[0]*
                                      test_targets.shape[1],test_targets.shape[2])),index=test_indices)

npwp_ti_reshape = npwp_ti.reshape(npwp_t.shape[0],npwp_t.shape[1])
ppwp_ti_reshape = ppwp_ti.reshape(ppwp_t.shape[0],ppwp_t.shape[1])

import matplotlib.lines as mlines
col_list = ['Pore Water Pressures - 63cm',
       'Pore Water Pressures - 113cm', 'Pore Water Pressures - 244cm']
fig, ax = plt.subplots(3,1,figsize=(8,10),sharex=True)
i = 1
for col in col_list:
    while i < (len(col_list)+1):
      ax1 = ax[i-1]

      for batch in range(0,npwp_t.shape[0],1):
        ax1.plot(npwp_ti_reshape[batch,:],npwp_t[batch,:,i-1], 
                 label='Training Data',c='lightslategray',linestyle='--',alpha=1)
        #ax1.plot(test_indices_reshape[batch,:],test_targets[batch,:,i-1], 
         #        label='test n',c='slategray',linestyle='--',alpha=1)
       
        #ax1.plot(test_indices_reshape[batch,:],npwp_predictions[batch,:,i-1], label='pred',c='b')
      ax1.scatter(all_target_indices,intervals,c='k',marker='|',s=35,alpha=0.5)

      for batch in range(0,ppwp_t.shape[0],1):
        ax1.plot(ppwp_ti_reshape[batch,:],ppwp_t[batch,:,i-1], label='test p',c='darkslategrey')
        ax1.plot(ppwp_ti_reshape[batch,:],ppwp_predictions[batch,:,i-1], label='pred p',c='indianred')
      ax1.set_xlim(pd.to_datetime('2012-01-14'),pd.to_datetime('2012-02-14'))
      for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
      ax1.set_ylim(-9,3.5)
      #ax1.set_ylabel('Suction (kPa)')

      
      color = 'royalblue'
      ax2 = ax1.twinx()
      #ax2.bar(ocr.index[:],ocr.iloc[:,1],width=0.2,color=color,alpha=0.7)
      ax2.tick_params(axis='y', labelcolor=color)
      ax2.set_ylim(0.8,14)
      #ax2.set_ylabel('Rainfall (mm)', color=color)
      ax2.axhline(y=4,color='k',linestyle='dotted',alpha=0.5)
      
      red_line = mlines.Line2D([], [], color='indianred', marker='',
                          markersize=15, label='Predictions')
      dashed_line = mlines.Line2D([], [], color='lightslategrey', marker='',
                          markersize=15, label='Training Data',linestyle='--')
      solid_line = mlines.Line2D([], [], color='darkslategrey', marker='',
                          markersize=15, label='Test Data',linestyle='-')
      vert = mlines.Line2D([], [], color='k', marker='|',
                          markersize=15, label='Prediction Interval (36hr)',
                           alpha=0.5)
      vert.set_linestyle('')

      fsize = 12
      plt.setp(ax1.get_xticklabels(), fontsize=fsize)
      plt.setp(ax1.get_yticklabels(), fontsize=fsize)
      plt.setp(ax2.get_yticklabels(), fontsize=fsize)

      if i ==1:
          plt.legend(handles=[red_line,dashed_line,solid_line,vert],loc='upper center',
                     fontsize='small',bbox_to_anchor=(0.51,1.01),ncol=4,
                     markerscale=0.45,frameon=False)
          #plt.title(('Soil Matric Potential'))
      
      
      i = i+1

#handles, labels = ax1.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center')
plt.subplots_adjust(hspace=0.1)
#plt.tight_layout()
#plt.savefig('36hr_48_d150_drop5_mask4mm_JF.svg',dpi=300)
plt.show()

