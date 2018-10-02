import WifiSensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

class WifiCNN:
    def __init__(self):
        self.totaldata = pd.DataFrame()

    def opendata(self):
        for x in range(3):
            print(x)
            ws = WifiSensor.WifiCollector()
            self.data = ws.get_multiple_values(10,0.5)
            self.datashape = self.data.shape
            height = np.zeros((self.datashape[0], 1))
            self.data["x"] = height+x*2
            self.data["y"] = height+x*5
            self.totaldata = self.totaldata.append(self.data)
        print(self.totaldata)
        self.totaldata = self.totaldata.fillna(0)/100
        self.X_data = self.totaldata.iloc[:, np.r_[0:self.datashape[1]]].as_matrix()
        self.y_data = self.totaldata.iloc[:, np.r_[self.datashape[1]:self.datashape[1]+2]].as_matrix()
        print(self.X_data)
        print(self.y_data)


    def cnn(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, test_size=0.3, random_state=234)
        model = Sequential()
        model.add(Dense(10, activation='tanh',input_shape=(self.X_data.shape[1],)))
        model.add(Dense(5, activation='tanh'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train,epochs=200,verbose=0)
        score = model.evaluate(X_train, y_train, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def cnn_data(self):
        print("helo")



if __name__ == '__main__':
    wcnn = WifiCNN()
    wcnn.opendata()
    wcnn.cnn()
    #wcnn.get_value_average()
    #for x in range(1):

