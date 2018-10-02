import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import wave
import pandas as pd
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

fs, data = wavfile.read('95328__ramas26__c.wav')
y= data[3000:93000]


n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(int(n/2))]
data = pd.DataFrame(frq,columns =['freq'])
data['values'] = abs(Y)

top_100 = data.sort_values('values', ascending=False).head(100)

print(top_100)
fig, ax = plt.subplots(2, 1)
ax[0].plot(y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum

ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
plt.show()
