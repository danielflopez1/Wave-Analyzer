import pywifi
import time
import pickle

class wifi:
    def __init__(self):
        self.state_dict = {}
        self.past_results = ''
        self.curr_state = 0
        self.wifi = pywifi.PyWiFi()
        self.iface = self.wifi.interfaces()[0]
        self.differences = 0
        self.old_bssds = []




    def check_differences(self,dict1, dict2, alignment):
        merged_dict = {**dict1, **dict2}
        for key in dict1:
            if key in dict2:
                align = [0] * (alignment - len(dict1[key]))  # align in case it is not found on this iteration or later
                #print(align+dict1[key],dict2[key])
                merged_dict[key] = align +dict1[key]+dict2[key]

        #print(merged_dict)
        return merged_dict

    def print_dict(self,signals):
        print("\n")
        for key in signals:
            print(key, signals[key])

    def get_data(self,state,alignment,number):
        #data = self.data
        self.iface.scan()
        bsses = self.iface.scan_results()
        if bsses in self.old_bssds:
            return number
        signals = {}
        for address in bsses:
            if (address.bssid in signals):
                continue
            else:
                signals[address.bssid] = [address.signal]

        if (state not in self.state_dict):
            self.state_dict[state] = signals
        else:
            self.state_dict[state] = self.check_differences(self.state_dict[state],signals,alignment)
        print(number,self.state_dict)
        self.old_bssds.append(bsses)
        return number+1


    def printing(self,state_dict):
        for device in state_dict:
            print(device, state_dict[device])


w = wifi()
alignment = {}

states = ["1","2","3","4","5","6","7","8","9","99"]
for state in states:
    x = 0
    while x <30:
        #print(state,x, alignment)
        #state = input('Give State: ')
        if (state == ''):
            continue
        if (state == '99'):
            with open('filesname3.pickle', 'wb') as handle:
                pickle.dump(w.state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #w.printing(w.state_dict)
            break
        if (state in alignment):
            alignment[state] = alignment[state] + 1
        else:
            alignment[state] = 0
        y = w.get_data(state, alignment[state],x)
        if(x==y):
            alignment[state] = alignment[state] - 1
        x = y
    print("Sleeping 5")
    time.sleep(10)
