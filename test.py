from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
import math
from matplotlib import pyplot as plt
from python_speech_features import mfcc
import os


#training start
#probability matrix for truck
prob_ma = [0 for x in range(605)]
prob_ma_nt = [0 for x in range(605)]


##for truck
#getting  the clip names
clip_names = []
directory = "."
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        clip_names.append(filename)

clip_number = 0
for clip in clip_names :
    #print (clip)
    check = [0 for c in range(605)]
    #def calc_fft(fname):
    fs_rate, signal = wavfile.read(clip)
    #print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    #print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    #print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    #print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    #print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    #FFT = abs(scipy.fft(signal))
    FFT = scipy.fft(signal)
    f = open("fs100.txt", "w+")
    #print (len(FFT))
    for i in FFT:
        f.write(str(i))
        f.write(",")
            #print str(abs(c[i]))
            #print "\n"
    f.close()
    
    abs_FFT = abs(FFT)[:701]
    for index,value in enumerate(abs_FFT) :
        if value >= 0.65*max(abs_FFT[100:]) and index>=100 and index<=700 :
            #print(clip_number, index)
            try :
                if check[(index)-100] != 1 :
                    prob_ma[(index)-100]+=1
                    check[(index)-100]=1
            except IndexError:
                print (clip_number,index)
                print ("Index Error")
    #print (prob_ma)
    clip_number+=1
#print (prob_ma)






##for non truck
##for truck
#getting  the clip names
clip_names = []
directory = "no_truck"
#directory = "."
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        clip_names.append("no_truck\\"+filename)
        #clip_names.append(filename)



clip_number = 0
for clip in clip_names :
    #print (clip)
    check = [0 for c in range(605)]
    #def calc_fft(fname):
    fs_rate, signal = wavfile.read(clip)
    #print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    #print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    #print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    #print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    #print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    #FFT = abs(scipy.fft(signal))
    FFT = scipy.fft(signal)
    f = open("fs100.txt", "w+")
    #print (len(FFT))
    for i in FFT:
        f.write(str(i))
        f.write(",")
            #print str(abs(c[i]))
            #print "\n"
    f.close()
    
    abs_FFT = abs(FFT)[:701]
    for index,value in enumerate(abs_FFT) :
        if value >= 0.65*max(abs_FFT[100:]) and index>=100 and index<=700 :
            #print(clip_number, index)
            try :
                if check[(index)-100] != 1 :
                    prob_ma_nt[(index)-100]+=1
                    check[(index)-100]=1
            except IndexError:
                print (clip_number,index)
                print ("Index Error")
    #print (prob_ma)
    clip_number+=1

    

#test starts here
#min_prob = 4.096439816370802e-11
#count = 0
div_nt = 70
div = 60
test_clips = []
directory = "test_truck"
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        test_clips.append('test_truck\\'+filename)
print (test_clips)

all_probs_truck = []
all_probs_non_truck = []

for clip in test_clips :
    truck_prob = math.log(0.5)
    nt_prob = math.log(0.5)
    #run time
    #def calc_fft(fname):
    fs_rate, signal = wavfile.read(clip)
    print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    #print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    #FFT = abs(scipy.fft(signal))
    FFT_T = scipy.fft(signal)

    abs_FFT_T = abs(FFT_T)[:701]

    for index,value in enumerate(abs_FFT_T) :
        if value >= 0.65*max(abs_FFT_T[100:]) and index>=100 and index<=700 :
            #for truck
            if prob_ma[(index)-100] == 0 :
                truck_prob = truck_prob + math.log(0.001)
            else :
                truck_prob = truck_prob + math.log(prob_ma[(index)-100]*1.0/div)
            #for non_truck
            if prob_ma_nt[(index)-100] == 0 :
                nt_prob = nt_prob + math.log(0.001)
            else :
                nt_prob = nt_prob + math.log(prob_ma_nt[(index)-100]*1.0/div_nt)
                
    deno = math.log( math.exp(truck_prob) + math.exp(nt_prob) )
    pt = math.exp(truck_prob - deno)
    pnt = math.exp(nt_prob - deno)
    all_probs_truck.append(pt)
    all_probs_non_truck.append(pnt)
    if pt > pnt :
        print ("Clip :" + clip + "is a Truck sound. P(T) :"+(str)(pt)+" P(NT) :"+(str)(pnt))
    else :
        print ("Clip :" + clip + "is a Non-Truck sound. P(T) :"+(str)(pt)+" P(NT) :"+(str)(pnt))

    
    '''
    if truck_prob < min_prob :
        print ("THIS IS NON TRUCK")
        print (clip , truck_prob)
        count +=1
    else :
        print ("THIS IS TRUCK")
        print (clip , truck_prob)
    '''    


    
#print (count)










#
# FFT_side = FFT[range(N/2)] # one side FFT range
# freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
# fft_freqs = np.array(freqs)
# freqs_side = freqs[range(N/2)] # one side frequency range
# fft_freqs_side = np.array(freqs_side)
# plt.subplot(311)
# p1 = plt.plot(t, signal, "g") # plotting the signal
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.subplot(312)
# p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Count dbl-sided')
# plt.subplot(313)
#
# # f = open(".txt", "w+")
# # for i in FFT_side:
# #     f.write(str(abs(i)))
# #     f.write(",")
# #         #print str(abs(c[i]))
# #         #print "\n"
# # f.close()
#
# p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Count single-sided')
# plt.show()
#
#
# # directory = "."
# # for filename in os.listdir(directory):
# #     if filename.endswith(".wav"):
# #         calc_fft(filename)
