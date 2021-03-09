import subprocess

data = '''b'Discovery started\nhci0 type 7 discovering on\nhci0 dev_found: 65:BD:2F:FF:F9:A2 type LE Random rssi -79 flags 0x0000 \nAD flags 0x1a \neir_len 19\nhci0 dev_found: 6A:9A:B6:4B:C8:CC type LE Random rssi -79 flags 0x0004\nAD flags 0x00\neir_len 31\nhci0 dev_found: 6B:42:76:E8:DC:2B type LE Random rssi -77 flags 0x0004\nAD flags 0x00 \neir_len 28\nhci0 dev_found: 43:1B:8B:EB:87:A6 type LE Random rssi -70 flags 0x0000\nAD flags 0x1a\neir_len 17\nhci0 dev_found: 40:43:82:91:AB:6F type LE Random rssi -64 flags 0x0004'''
device_dict = {}
alignment = 0
def wordModifications(results,state):
    global alignment
    index = 0

    device_size = len('65:BD:2F:FF:F9:A2')
    signal_size = len(' 6A:9A:B6:4B:C8:CC type LE Random rssi ')
    min_dict = []
    while True:
        index = results.find(':', index)+2
        if(index == 1):
            break
        device = results[index:index + device_size]
        signal = results[index+signal_size:index+signal_size+2] + ' '+str(state)+ ' '+ str(alignment)
        if(device in min_dict):
            index = index + signal_size
            print("Skipped",device,signal)
            continue
        min_dict.append(device)
        if(device in device_dict):
            device_dict[device].append(signal)
        else:
            device_dict[device]= [signal]
        index = index+signal_size
    print(len(device_dict))
    alignment = alignment + 1
    print(device_dict)


def get_data(state):
    test = subprocess.Popen(["sudo",'-S', "btmgmt", "find", "rssi"], stdout=subprocess.PIPE)
    data = test.communicate()[0]
    #print(data)
    print(wordModifications(str(data),state))


def get_state():
    return input('Give State')


while True:
    get_data(get_state())




