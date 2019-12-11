# Wave Analyzer

This project uses a Raspberry Pi to acquiring WiFi signals, analyze the signals and use them to train a neural network for object obstruction.
I started by aquiring the signals, get the average strength of the signals with a moving average and select the 10 strongest signal strengths. Use the strength of the signals to train the network by comparing them to an object coordinate or state. This can be done by using MLP with input signals of specific states with a series of 10 signals over 10 seconds (100 inputs) or CNN by setting the signals in parallel 10x10. 
 
## Prerequisites

This project uses Tensorflow and Keras

## Raspberry Pi
Raspbian Kernel:
```
version:4.15<
```

## Warnings
Note the networks have changed to be used on other datafiles not available for copyright reasons.
