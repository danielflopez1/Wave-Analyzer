import pywifi
import time
import pickle

class wifi:
    def __init__(self):
        self.state_dict = {}
        alignment = {}
        self.past_results = ''
        self.curr_state = 0
        self.wifi = pywifi.PyWiFi()
        self.iface = self.wifi.interfaces()[0]
        while True:
            state = input('Give State: ')

            print(state)
            if(state == ''):
                continue
            if (state == '99'):
                with open('filesname2.pickle', 'wb') as handle:
                    pickle.dump(self.state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.printing(self.state_dict)
                break
            if (state in alignment):
                alignment[state] =alignment[state]+1
            else:
                alignment[state] = 0
            print(self.get_data(state,alignment[state]),'\n',alignment)



    def check_differences(self,dict1, dict2, alignment):
        merged_dict = {**dict1, **dict2}
        for key in dict1:
            if key in dict2:
                align = [0] * (alignment - len(dict1[key]))  # align in case it is not found on this iteration or later
                print(align+dict1[key],dict2[key])
                merged_dict[key] = align +dict1[key]+dict2[key]

        print(merged_dict)
        return merged_dict

    def print_dict(self,signals):
        print("\n")
        for key in signals:
            print(key, signals[key])

    def get_data(self,state,alignment):
        #data = self.data
        self.iface.scan()
        time.sleep(1)
        bsses = self.iface.scan_results()
        print(bsses[0].bssid)
        signals = {}
        for wifi in bsses:
            if (wifi.bssid in signals):
                continue
            else:
                signals[wifi.bssid] = [wifi.signal]

        if (state not in self.state_dict):
            self.state_dict[state] = signals
        else:
            self.state_dict[state] = self.check_differences(self.state_dict[state],signals,alignment)
        return self.state_dict


    def printing(self,state_dict):
        for device in state_dict:
            print(device, state_dict[device])

w = wifi()