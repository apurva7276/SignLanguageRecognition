import os
import numpy as np
from keras.models import load_model
import Required_variables as rv

# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we are try to detect
actions = rv.get_actions()
print(actions)

X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
model = load_model("action_v4.h5")
res = model.predict(X_test)

print(res)

print(actions[np.argmax(res[0])])
print(actions[np.argmax(y_test[0])])

