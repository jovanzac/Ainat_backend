import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from Utils.support_functions import SupportFunctions

class LSTMModel :
    def __init__(self, *args, **kwargs) :
        self.log_dir = os.path.join(os.getcwd(),'Logs')

        support.create_dir(self.log_dir)
        tb_callback = TensorBoard(log_dir=self.log_dir)
        
    def define_lstm_model(self) :
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=support.neural_inp_shape))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(support.actions.shape[0], activation='softmax'))
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy','categorical_crossentropy'])
        
        return model
    
    
    def preprocess_data_for_model(self) :
        sequences, labels = support.load_keypoints()
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        
        return X_train, X_test, y_train, y_test
        
    
    def train_model(self, X_train, y_train, loc) :
        model = self.define_lstm_model()
        model.fit(X_train, y_train, epochs=250, callbacks=[callback])
        model.save(loc)
        

    def load_lstm_model(self, loc) :
        saved_model = load_model(loc)
        
        return saved_model


class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["categorical_accuracy"]
        if accuracy >= self.threshold:
            self.model.stop_training = True
            

callback=MyThresholdCallback(threshold=0.95)

support = SupportFunctions()
