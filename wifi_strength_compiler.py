import subprocess
import re
import time
import pandas as pd
import pickle
import numpy as np

class wifi:
    def __init__(self):
        self.data = b'BSSID              SIGNAL \nFA:8F:CA:87:57:97  85     \n74:83:C2:D3:4A:27  72     \n76:83:C2:A3:4A:27  70     \n76:83:C2:93:4A:27  65     \nFA:8F:CA:64:48:24  64     \n00:1F:01:34:2A:94  54     \n96:AC:B9:B4:2D:67  54     \n74:AC:B9:B4:2D:67  54     \n86:AC:B9:B4:2D:67  47     \n84:94:8C:55:DD:38  35     \nD0:60:8C:03:93:28  30     \n86:94:8C:55:DD:30  29     \n02:71:47:05:04:1C  27     \n54:64:D9:EF:83:F5  17     \n'
        self.state_dict = {}
        self.alignment = 0
        self.past_results = ''
        self.curr_state = 0
        while True:
            state = self.get_state()
            print(state)
            if(state == ''):
                continue
            if (state == '99'):
                with open('filesname21'
                          '1.pickle', 'wb') as handle:
                    pickle.dump(self.state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.printing(self.state_dict)
                break
            print(self.get_data(state))
            time.sleep(1)


    def check_differences(self,dict1, dict2):
        merged_dict = {**dict1, **dict2}
        for key in dict1:
            if key in dict2:
                print(dict1[key],dict2[key])
                merged_dict[key] = dict1[key]+dict2[key]
        print(merged_dict)
        return merged_dict


    def wordModifications(self,results,state):
        results = results.replace('Mbit/s','')
        results = re.sub(' +', ' ', results)
        results = results.split('\\n')
        print(results)

        min_dict = {}
        for line in results[1:-1]:
            data = line.split(' ')[:-1]
            #print(data)
            min_dict[data[0]] = [int(data[1])]
        #print(len(state_dict))
        if(state not in self.state_dict):
            self.state_dict[state] = min_dict
        else:
            self.state_dict[state] = self.check_differences(self.state_dict[state],min_dict)
        print(self.state_dict)
        return self.state_dict

    def get_data(self,state):
        #data = self.data
        data = str(subprocess.check_output(["nmcli","-f", "BSSID,SIGNAL", "dev", "wifi"]))
        #data = str(data)
        #print(data)
        return self.wordModifications(str(data),state)


    def get_state(self):
        return input('Give State: ')

    def printing(self,state_dict):
        for device in state_dict:
            print(device, state_dict[device])

w = wifi()