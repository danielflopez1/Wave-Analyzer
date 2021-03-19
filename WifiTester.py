import subprocess
import re
import time
import pandas as pd
import pickle
import numpy as np

def wordModifications(results):
    results = results.replace('Mbit/s', '')
    results = re.sub(' +', ' ', results)
    results = results.split('\\n')
    print(results)
    mydata = []
    for line in results[1:-1]:
        data = line.split(' ')[:-1]
        mydata.append(data)

    return mydata

def get_pickle():
    full_data = {}
    with open('filesname211.pickle', 'rb') as handle:
        full_data = pickle.load(handle)
        print(full_data)
    return full_data

def activation(list, datum):
    #print(list[-3:], datum)
    mymax = np.amax(list[:])
    mymin = np.amin(list[:])
    aver = np.average(list[:])
    offset = 3
    if(datum >mymin and datum < mymax):
        return 1
    else:
        return 0


def raw_check(init_dict, signal):
    right_key = 0
    num_total = len(signal)
    print(num_total)
    for key in init_dict:
        right_key = 0
        for data in signal:
            if(data[0] in init_dict[key]):
                main_data = init_dict[key][data[0]]
                new_data = int(data[1])
                right_key = right_key+activation(main_data,new_data)
        print("key:", key,"key:" ,right_key)





def get_data():
    #global data
    data = str(subprocess.check_output(["nmcli","-f", "BSSID,SIGNAL", "dev", "wifi"]))
    #data = str(data)

    return wordModifications(str(data))

pickle_data = get_pickle()
print(pickle_data['1'])
for datums in get_data():
    print(datums)
#print(get_data())
raw_check(pickle_data,get_data())
#get_data(1)