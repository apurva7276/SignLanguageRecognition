import numpy as np
import os

# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('MP_Data3')

#name of the latest model
model_name= 'action_v7.h5'

# Actions that we are try to detect
#actions = np.array(['hello', 'thankyou', 'welcome', 'airplane', 'bye', 'back', 'yes', 'cancel', 'beautiful', 'center','name'])
#actions = np.array(['finish', 'first', 'hey', 'excuseme', 'student', 'understand', 'no', 'excited', 'drink', 'equal'])
actions = np.array([])
actions_list = []
def get_actions():
    return actions

def get_modelName():
    return model_name

def detect_actions_in_folder():
    actions_list=[]
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH))):
        if os.path.isdir(os.path.join(DATA_PATH,sequence)):
            print(sequence)
            actions_list.append(str(sequence))

    actions =np.array(actions_list)
    print(actions.shape[0])
    return actions


detect_actions_in_folder()
