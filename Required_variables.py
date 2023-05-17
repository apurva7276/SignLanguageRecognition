import numpy as np
import os

# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we are try to detect
actions = np.array(['hello', 'thankyou', 'welcome', 'airplane', 'bye', 'back', 'yes', 'cancel', 'beautiful', 'center','name'])

def get_actions():
    return actions