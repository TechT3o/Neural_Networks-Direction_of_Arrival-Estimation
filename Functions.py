import wave
import numpy as np
import webrtcvad
import pyaudio
import scipy.io
import tflite_runtime.interpreter as tflite
from pyargus import directionEstimation as pa

def npow2 (x) :
# finds the next power of 2 of the input x
	return 1 <<(x -1) . bit_length ()

def stftprep (x , fs , time ):
# splits the input array in an array of frames of fixed length (to be used to prepare
#array for VAD or STFT )
#x = input array
#fs = sampling frequency
# time = time of the split frames ( determines the frame length )

# returns the array of the split input data
   framesamp = int( time * fs )
   X = np . array ([ x[i: i+ framesamp ] for i in range (0 , len (x) - framesamp , framesamp ) ])
   return X
def VAD ( aggr , data ,sr , time ):
# aggr how sensitive voice detection is
# data ( numpy array ) to be classified as voiced / unvoiced
#sr is the sampling rate
# the time of the frame in ms (10 ,20 ,30 ms)

   # Sets the vad object
   vad = webrtcvad . Vad ( aggr )
   # sets frame length
   framesz = time /1000
   # cuts the data in the frame size
   Z = stftprep ( data , sr , time /1000)
   # returns indexes of voiced data
   ind = [i for i in range ( len (Z) ) if vad . is_speech ( np . int16 ( Z[i ]*32768) . tobytes () ,sr )]
   # multiply by 32768
   # creates the array of the voiced parts of the data
   emin = [Z [k] for k in ind ]
   eminor = np . array ([ j for i in emin for j in i ])
   return eminor

def pyrec (fs , duration , channels =1) :
# Function that records multi channel audio using pyaudio

#fs = sampling frequency
# duration = recording duration
# channels = number of microphone inputs ( defaults to 1)

    CHUNKSIZE = int( fs * duration ) # fixed chunk size depending on sampling frequency and recording duration

    # open stream
    p = pyaudio . PyAudio ()
    stream = p . open ( format = pyaudio . paInt16 , channels = channels , rate = fs , input = True , frames_per_buffer = CHUNKSIZE )

    # read the microphone input calues and modify in an channelsXchunksize shape numpy array
    data = stream . read ( CHUNKSIZE )
    numpydata = np . frombuffer ( data , dtype = np . int16 )
    numpydata = np . reshape ( numpydata , ( CHUNKSIZE , channels ) )
    numpydata = [ numpydata [: , i] for i in range ( channels )]

    # close stream
    stream . stop_stream ()
    stream . close ()
    p. terminate ()
    return numpydata

def gcc_phat ( sig , refsig , fs , max_tau = None , interp =4) :
    '''
    I modified this function from the original :
    Copyright (c) 2017 Yihui Xiong
    Licensed under the Apache License , Version 2.0 ( the " License ");
    you may not use this file except in compliance with the License .
    You may obtain a copy of the License at
    http :// www . apache . org/ licenses / LICENSE -2.0
    '''
    
    #This function computes the offset between the signal sig and the reference signal refsig
    #using the Generalized Cross Correlation - Phase Transform (GCC - PHAT ) method.

    # make sure the length for the FFT is larger or equal than len ( sig ) + len ( refsig )
    n = sig . shape [0] + refsig . shape [0]

    # find next power of 2 to make FFT faster
    n = npow2 ( n)

    # Generalized Cross Correlation Phase Transform
    # Uncomment to add window , multiply the signals sig and refsig with the window
    # window = np. hamming ( len ( sig ))

    SIG = np . fft . rfft ( sig , n =n)
    REFSIG = np . fft . rfft ( refsig , n=n)

    R = SIG * np . conj ( REFSIG )
    # phat weighting
    ab = np . abs ( R)
    ab [ ab < 1e-10] = 1e-10
    # obtains gcc
    cc = np . fft . irfft (R / np . abs ( ab ) , n = npow2 ( interp * n))

    # Uncomment below to normalize
    #cc = cc/np. max (cc)

    max_shift = int( interp * n / 2)
    if max_tau :
        max_shift = np . minimum ( int ( interp * fs * max_tau ) , max_shift )
    
    cc = np . concatenate (( cc [ - max_shift :] , cc [: max_shift +1]) )
    # find max cross correlation index ( TDOA )
    shift = np . argmax ( np . abs ( cc ) ) - max_shift

    tau = shift / float ( interp * fs )

    return tau , cc
def myinwhole ( Dat ,fs , mics = 6) :
    # function that generates the GCC - PHAT function for all different microphone pair combinations

    # Dat = input audio data ( data must be of shape mics X any audio length )
    ind = list ( itertools . combinations ( range ( mics ) ,2) )
    GCC = []

    for i ,j in ind :
        t ,c = gcc_phat ( Dat [i] , Dat [j ], fs , 0.0003)

    GCC . append (c)

    GCC = np . array ( GCC )
    GCC = np . transpose ( GCC )
    # Change the 4 below with the interpolation factor given in gccphat () function
    G = -len (c ) /(2*4* fs )
    #X and Y are included to be able to print the input image with matplotlib.pyplot.colormesh (X,Y, GCC )
    Y = np . linspace (G , -G , len (c ) +1)
    X = np . linspace (1 , len( ind ) ,len ( ind ) +1)
    return X , Y , GCC

def power (a , fs ):
    # function that finds which element in an array has the max power and returns its index

    #a = array
    #fs = sampling frequency
    P = []
    for i in range ( len (a )):
        x = sum ( np.square (a[i]))* fs / len (a[i])
    P. append (x )
    ind = np . argmax ( P)
    return ind

def mic_index (i , n =6) :
    # This function reorders the channels with respect to the reference microphone

    # i is the index of the microphone with the maximum power
    # n is the number of microphones in my mic array
    # ind has all the different possible mic combinations ( the order matters as the neural network inputs are generated in this sequence )
    ind = np . array ( list ( itertools . combinations ( range (n) ,2) ))
    fixed = ( ind + i)%n # modulus operation wiht base 6 returns the mic indexes rotated so that mic with max power is at position of 0
    return fixed

def beamform (a ,k , mat ):
    # This function performs phase shift beamforming to a multichannel audio input

    #a = audio input
    #k is the column that has the direction in the beamforming weights matrix
    # mat = beamforming weights matrix of shape kXchannels can be obtained from MATLAB and imported as
    # mat = scipy .io. loadmat (’ beamformerwt1 . mat ’)
    # mat = mat [’wt ’]

    Y = 0
    for i in range ( len (a )):
        # finds frequency spectrum of audio signal for each microhpone
        A = np . fft . rfft (a[ i ])
        # selects the weights for each microphone in direction k
        w = mat [i , k]
        # multiplies frequency spectrum by complex beamformign weights
        x = A* w
        # Sums to get the frequency spectrum of the beamformed signal
        Y += x
    # retrieve the beamformed signal in time
    #(This step can be skipped and return Y instead to find the maximum energy according to Parsevals theorem )
    y = np . fft . irfft (Y )
    return y

def order (a) :
    # reorders the microphone inputs to match the simulated data
    #( This is inlcuded because the channel ordering of the Respeaker is clockwise
    # while the simulated channel order by librosa is counter clockwise )
    o = [a [0] , a [5] , a [4] , a [3] , a [2] , a [1]]
    return o

def setinterpreter ( path , inp ):
    # sets the tflite interpreter so that the model can make its prediction , passes the input to the model and returns the prediction

    # path is the path to the . tflite model
    # inp = input on which the prediction should be made (gcc - phat matrix of myinwhole () function )

    interpreter = tflite . Interpreter ( model_path = path )
    interpreter . allocate_tensors ()
    input_details = interpreter . get_input_details ()
    output_details = interpreter . get_output_details ()
    input_shape = input_details [0]["shape"]
    interpreter.set_tensor(input_details [0]["index"],inp )
    interpreter.invoke ()
    output_data = interpreter.get_tensor ( output_details [0]["index"])
    return output_data

def doaconv(i,j):
    #This function converts the output of the Neural Network and the beamformer in the 60-degree implementation
    #into the direction of arrival in 360 degrees
     
    #i is the output of the beamformer
    #j is the output of the neural network
    res = 10
    doa  = np.arange(-25,25+res,res)
    off = i * 60
    deg = off + doa[j]
    return deg

def doaconv360(j):
    #This function converts the output of the Neural Network in the 360-degree implementation
    #into the direction of arrival in 360 degrees

    #j is the output of the neural network
    doa  = np.arange(0,360,10)
    deg = doa[j]
    return deg

def music(rec):
    #Function that performs the pyargus  module MUSIC algorithm for DOA

    #rec is the received audio signal

    #Pyargus MUSIC performs DOA in 180 degrees so beamforming is used to make it 360
    mat = scipy.io.loadmat('beamformerwt1.mat')
    mat = mat['wt']

    fs = 16000
    d = 0.03
    channels = 6

    rec = np.array(rec)
    r = 0.0463
    y = [beamform(rec,i,mat) for i in range(6)]
    i = power(y,fs)

    #Pyargus commands that do MUSIC with 10 degrees resolution
    R = pa.corr_matrix_estimate(rec.T, imp = "mem_eff")
    deg = np.arange(0,181,10)
    sv = pa.gen_uca_scanning_vectors(channels,r,deg)
    #sd = pa.estimate_sig_dim(R)
    MUSIC = pa.DOA_MUSIC(R,sv,1,10)
    direction = np.argmax(MUSIC)
    direction = direction*10

    #This part converts the MUSIC 180 degrees reading to 360 degrees
    if i == 0 or i == 1 or i == 5 :
        if direction > 90:
            direction = 180-direction
        else:
            direction = 360-direction
    else:
        
        if direction == 180:
            direction = 180
        elif direction <= 90:
            direction = 180-direction
        else:
            direction = 360 - direction
    return direction


def recnsaveWAV(filename, fs = 16000, seconds = 0.15, channels = 6, th=500):
    #Function records an audio segment, performs VAD with an energy threshold, saves the recorded audio if it is voiced and returns the numpy array of the audio and a conditional if it is is voiced

    #Filename is the path where the audio is to be saved
    #fs is the sampling rate
    # seconds is the audio frame length
    #channels is the number of audio channels
    # th is the energy threshold above which the segment is voiced

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    CHUNKSIZE = int(fs*seconds) # fixed chunk size

    #This part records the audio segment
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    
    stream = p.open(format=sample_format,
                            channels=channels,
                            rate=fs,
                            frames_per_buffer=chunk,
                            input=True)


    data = stream.read(CHUNKSIZE)
    
    #converts audio bytes to int16 numpy array format
    numpydata = np.frombuffer(data, dtype=np.int16)
    numpydata = np.reshape(numpydata, (CHUNKSIZE, channels))
    numpydata = [numpydata[:,i] for i in range(channels)]
 
    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
 
    #VAD with energy threshold   
    if np.sum(np.square(numpydata[0]))/len(numpydata[0]) > th and np.sum(np.square(numpydata[1]))/len(numpydata[0]) > th and np.sum(np.square(numpydata[2]))/len(numpydata[0]) > th and np.sum(np.square(numpydata[3]))/len(numpydata[0]) > th and np.sum(np.square(numpydata[4]))/len(numpydata[0]) > th and np.sum(np.square(numpydata[5]))/len(numpydata[0]) > th:

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(data)
        wf.close()
        return numpydata , True
    else:
        return numpydata, False