from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import Required_variables as rv

# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we are try to detect
#actions = np.array(['hello', 'thanks', 'iloveyou'])
actions = rv.get_actions()

def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    filepath = "action.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def train_model():
    model, callback_list = create_model()

    #load data created using Preprocess_data.py
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))

    print(X_train)
    print(y_train)

    model.fit(X_train, y_train, epochs=100, callbacks=[callback_list])

    model.summary()

    model.save('action.h5')

train_model()




