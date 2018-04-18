# ## Trigger Word Detection Assistant
# 
# Now there can be various different situations in which the trigger words can be spoken, in order to make the model better at detecting those words it is important to mimic the situations where there can be lot of background noise. For that we are going to overlap speech audio files with background noise audio files so as to create that effect.<br>
# The main reason behind not using audio in actual noisy environment is because during labelling the Y labels for the timesteps we need to know where the **"activate"** word ends , now using synthesised data we will be knowing exactly where the word **"activate"** ends and makes labelling easier. 


import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from utility import *
import subprocess

# Loading necessary Keras library modules
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

# Model Dev Set Performance
from sklearn.metrics import f1_score

# for taking voice input from user
from voice_input import take_input

# ## Data preparation
# We will overlay the various speech audio clips on top of background audio files. For doing that it needs to be ensured that there is no overlap of audio clips with already overlayed audio clips on the background. Then we label the audio data , we make the Yi for the timestep when the positive audio clip ends "1" and also make the next 50 timesteps as "1" also, so that the model can detect the presence of "activate" in very short duration also.<br>
# The basic strategy will be:
# 1. Generate a random time segment.
# 2. Check if the generated segment causes an overlap or not for the new audio clip. If overlap happens then again generate a new segment, if not then add the new audio clip on top of background and add this time segment to the list of overlayed time segments.
# 3. Label the output Y labels by adding 1s.
# NOTE: We are going to use a pretrained model, so the code for creating training examples is not 
# mentioned here, that code can be found in "training.py" file


# ## Model Architecture
# **Step 1**: We will use a 1D CONV layer for extracting low level features from the audio file. This also makes our output smaller in dimension.
#
# We will be using two GRU layers.
#
# **Step 2**: First GRU layer. Output from this layer is passed onto the next GRU layer, but before that it goes through batch Normalization and Dropout layers.
#
# **Step 3**: Second GRU layer.
#
# **Step 4**: We create a time-distributed dense layer. Reason behind using TimeDistributed Layer is that we want the output from each timestep and pass it forward, otherwise we end up with only the final timestep output.
# We have used sigmoid activation since the output can be 0 or 1 only.
def create_model(input_shape):
    """
    for creating Keras Model graph.
    
    Arguments:
    input_shape -- shape of the model's input data 

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer : for extracting features
    X = Conv1D(filters = 196, kernel_size = 15, strides=4)(X_input)         # CONV1D
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)        # Batch normalization
    X = Activation('relu')(X)                                               # ReLu activation
    X = Dropout(0.8)(X)                                                     # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)                     # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                                  # dropout (use 0.8)
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)     # Batch normalization
    
    # Step 3: Second GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)                 # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                              # dropout (use 0.8)
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization
    X = Dropout(0.8)(X)                                               # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer 
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)        # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

# ## Making Predictions
# 1. Compute the spectrogram for the given audio file.
# 2. Using `np.swap` and `np.expand_dims` for reshaping the input to size (1, Tx, n_freqs)
# 5. Using forward propagation find the model predictions for each time step.
def detect_triggerword(filename, model):
    '''
    For detecting the presence of trigger words in the audio file
    
    Arguments:
    filename -- name of input audio file 
    model -- Keras Model instance

    Returns:
    predictions -- output timestep predictions
    '''
    plt.subplot(2, 1, 1)
    # for generating the spectogram
    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('Probability')
    plt.savefig('output//audio.png')
    return predictions


def action_on_activate(chime_file, filename, predictions, threshold):
    '''
    For detecting the presence of trigger words in the audio file and executing the required action
    
    Arguments:
    chime_file -- name of Notification alert audio file 
    filename -- name of input audio file 
    predictions -- output timestep predictions
    threshold --  for adjusting how precisly we want to detect the trigger word
    
    Returns:
        None
    '''
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
            print('Detected Trigger Word\n')
            # run the command that you want
            subprocess.call(r'C:\Program Files (x86)\Google\Chrome\Application\Chrome.exe')
        
    audio_clip.export("output//detected_output.wav", format='wav')


# Preprocess the audio to the correct format
def preprocess_audio(filename):
    '''
    For detecting the presence of trigger words in the audio file
    
    Arguments:
    filename -- name of input audio file 

    Returns:
        None
    '''
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')


def main():
    # ## Dataset
    # The audio data used in this model has been sampled at 44100 Hz i.e, that many numbers are produced each second to represent the audio. But for the model we are going to use the **spectogram** of the audio and use 5511 timesteps. We will be using **10 seconds** for each clip for the model. That means we are going to divide the 10seconds time interval into 5511 intervals. Incase the audio file is shorter it will be padded or if longer it will be truncated.<br> **Tx = 5511**. <br>
    # We will use 1375 timesteps for output.<br>
    # **Ty = 1375**.<br>
    # A spectogram is a graphical representation between the timesteps and frequency.

    # input timesteps
    Tx = 5511
    # output timesteps
    Ty = 1375
    # No. of frequencies input to the model at each timestep
    n_freq = 101

    model = create_model(input_shape=(Tx, n_freq))
    #  Model Overview
    model.summary()
    # Loading a pretrained model using 4000 examples since Trigger word detection takes a long time to train.
    model = load_model('./models/tr_model.h5')

    '''  # evaluate the model performance
    X_dev = np.load("./XY_dev/X_dev.npy")
    Y_dev = np.load("./XY_dev/Y_dev.npy")
    loss, acc = model.evaluate(X_dev, Y_dev)

    # reshapping
    Y_pred = model.predict(X_dev)
    Y_pred = Y_pred.reshape(Y_pred.shape[0], -1)
    Y_pred = Y_pred[:, :] >= 0.5
    Y_dev = Y_dev.reshape(Y_dev.shape[0], -1)

    f1 = f1_score(Y_dev, Y_pred, average='micro')
    print("Dev set accuracy = ", acc)
    print("Dev set F1 score = ", f1) '''

    # After finding the output predictions for each timestep of Ty, we need to insert the action that we want to perform when the word "activate" ends. For that we see in the output when 1 appears and then insert the action there and skip doing this for the next 75 timesteps since the next 75 timesteps are 1s only.
    chime_file = "audio//half.wav"

    # clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # record the user audio and return the saved audio filename
    # We record the audio till 10 seconds but rather for a shorter time and then pad,
    # since when the user wants the assistant to launch something then he/she should not wait for 10 seeconds.
    file_name = "audio//"
    file_name = file_name + take_input()

    # preprocess the recorded audio file
    preprocess_audio(file_name)
   
    # for adjusting how precisly we want to detect the trigger word
    alert_threshold = 0.5
    prediction = detect_triggerword(file_name, model)
    action_on_activate(chime_file, file_name, prediction, alert_threshold)
    

if __name__== "__main__":
    main()


# ### Credits: 
# https://www.coursera.org/learn/nlp-sequence-models/home/welcome
