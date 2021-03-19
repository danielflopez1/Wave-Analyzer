import subprocess
import re
import operator
import time
import collections
import numpy as np
import pandas as pd50200

class WifiCollector:
    def __init__(self):
        self.find_words = ['BSSID','Signal']
        self.offset = [1]
        self.untrusted_networks = []
        self.trusted_networks = []
        self.trusted_values = []
        self.network_dependability_avg = {}
        self.moving_avg = 100
        self.min_reliability= self.moving_avg*0.4


    def get_data(self):
        #self.results = str(subprocess.check_output(["netsh", "wlan", "show", "network", "mode=Bssid"])) #Windows
        self.results = str(subprocess.check_output(["nmcli","-f", "BSSID,RATE,SIGNAL", "dev", "wifi"]))

        print(self.results)


    def wordModifications(self):
        result = self.results.replace(':', '')
        result = result.replace(r'\r\n', '')
        result = result.replace('%', '')
        return result.replace('Basic rates (Mbps)', 'brmbps').split()


    def parse(self):
        self.get_data()
        result = self.wordModifications()
        #print(result)
        for i,words in enumerate(result):
            for keywords in self.find_words:
                if words in keywords:
                    print(words,"|", result[i+1])
        print()

    def save_data(self):
        self.get_data()
        result = self.wordModifications()
        #print(result)
        for i, words in enumerate(result):
            if words == self.find_words[0]:
                if(len(result[i+1])<3):
                    if(result[i + 2] not in self.untrusted_networks):
                        self.trusted_networks.append(result[i + 2])
                        self.trusted_values.append(result[i + 4])
                else:
                    if (result[i + 1] not in self.untrusted_networks):
                        self.trusted_networks.append(result[i + 2])
                        self.trusted_values.append(result[i + 3])
        #print(self.trusted_networks)

    def new_signals(self):
        self.get_data()
        result = self.wordModifications()
        #print(result)
        trusted_Networks = []
        trusted_Values = []
        for i, words in enumerate(result):
            if words == self.find_words[0]:
                if (len(result[i + 1]) < 3):
                    if (result[i + 2] not in self.untrusted_networks):
                        trusted_Networks.append(result[i + 2])
                        trusted_Values.append(result[i + 4])
                else:
                    if (result[i + 1] not in self.untrusted_networks):
                        trusted_Networks.append(result[i + 1])
                        trusted_Values.append(result[i + 3])
        #print(trusted_Networks)
        #print(trusted_Values)
        return trusted_Networks,trusted_Values

    def check_trusted_networks(self):
        network_dependability = {}
        untrusted_networks = []
        for x in range(self.moving_avg):
            time.sleep(0.3)
            nsig,nval = self.new_signals()
            nval = list(map(int,nval))
            #print(nsig,nval)
            for i, key in enumerate(nsig):
                if(key in network_dependability):
                    #print(key,network_dependability[key],nval[i])
                    network_dependability[key] = [network_dependability[key][0]+1, network_dependability[key][1]+nval[i]]
                else:
                    network_dependability[key] = [1,nval[i]]
        ordered_networks = collections.OrderedDict(sorted(network_dependability.items()))
        return ordered_networks

    def get_all_networks(self, num_timeframes, sleep_time):
        network_dependability = {}
        for x in range(num_timeframes):
            zeroappend = []
            for y in range(x):
                zeroappend.append(0)
            # print(zeroappend)
            time.sleep(sleep_time)
            nsig, nval = self.new_signals()
            nval = list(map(int, nval))

    def get_multiple_values(self,num_timeframes,sleep_time):
        network_dependability = {}
        for x in range(num_timeframes):
            zeroappend = []
            for y in range(x):
                zeroappend.append(0)
            #print(zeroappend)
            time.sleep(sleep_time)
            nsig, nval = self.new_signals()
            nval = list(map(int, nval))
            # print(nsig,nval)
            for i, key in enumerate(nsig):
                if (key in network_dependability):
                    # print(key,network_dependability[key],nval[i])
                    network_dependability[key].append(nval[i])
                else:
                    network_dependability[key] = zeroappend+[nval[i]]

            for key in network_dependability:
                if(key not in nsig):
                    network_dependability[key].append(0)

        ordered_networks = collections.OrderedDict(sorted(network_dependability.items()))
        #for network in ordered_networks:
        #    print(network,len(ordered_networks[network]),ordered_networks[network])
        network_df = pd.DataFrame.from_dict(ordered_networks)
        return network_df
    def get_value_average(self):
        avgNetwork = {}
        dependable_networks = self.check_trusted_networks()
        for network in dependable_networks:
            avgNetwork[network] = dependable_networks[network][1]/dependable_networks[network][0]

        return avgNetwork
        #for network in dependable_networks:
        #    self.network_dependability[network] = dependable_networks[network][1]/float(dependable_networks[network][0])
        #print(self.network_dependability)

if __name__ == '__main__':
    wc = WifiCollector()
    #avg_network = wc.get_multiple_values(10,0.2)
    for x in range(5):
        wc.get_data()
        time.sleep(1)
