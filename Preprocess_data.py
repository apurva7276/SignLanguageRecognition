from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import os
import Required_variables as rv

# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we are try to detect
#actions = np.array(['hello', 'thanks', 'iloveyou'])
actions = rv.get_actions()

# Thirty sequences worth of data for each action
no_of_sequences = 30

# {'hello': 0, 'thanks': 1, 'iloveyou': 2}
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(no_of_sequences):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = np_utils.to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# Saves the test and train split data into .npy files
# These files are accessed from Preprocess_data.py
def save(name, np_array):
    npy_path = os.path.join(DATA_PATH, name)
    np.save(npy_path, np_array)


save("X_train", X_train)
save("X_test", X_test)
save("y_train", y_train)
save("y_test", y_test)
