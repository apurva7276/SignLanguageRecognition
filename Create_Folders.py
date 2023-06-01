import os
import numpy as np
import Required_variables as rv

# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('MP_Data2')

# Actions that we are try to detect
#actions = np.array(['hello', 'thanks', 'iloveyou'])
actions = rv.get_actions()

# Thirty sequences worth of data for each action
no_of_sequences = 30

# Each sequence is going to be 30 frames in length
no_of_frames_per_video = 30

# Folder start
start_folder = 30
def create():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    for action in actions:
        folder_dir = os.path.realpath(os.path.join(DATA_PATH, action, ''))
        if not os.path.exists(folder_dir):
            # Create a new directory because it does not exist
            os.makedirs(folder_dir)

        #dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        for sequence in range(1, no_of_sequences+1):
            path = os.path.realpath(os.path.join(DATA_PATH+'/'+action+'/'+str(sequence), ''))
            if not os.path.exists(path):
                os.makedirs(path)


create()