import os
import numpy as np


class SupportFunctions :
    def __init__(self) :
        # Path for exported data, numpy arrays
        self.DATA_PATH = os.path.join(os.getcwd(),'MP_Data') 

        # Actions that we try to detect
        self.actions = np.array(["drowning", "normal"])

        self.create_dir(os.path.join(self.DATA_PATH))
        # Thirty videos worth of data
        self.no_sequences = 6

        # Videos are going to be 80 frames in length
        self.sequence_length = 80

        # Folder start
        self.start_folder = 0
        
        self.label_map = {label:num for num, label in enumerate(self.actions)}
        
        self.neural_inp_shape = (80, 306)
        
    
    def create_dir(self, path) :
        try: 
            os.makedirs(path)
        except FileExistsError :
            pass
        
    
    def create_dir_for_keypoints(self) :
        # Create a directory for each action
        for action in self.actions: 
            self.create_dir(os.path.join(self.DATA_PATH,action))

            # Creating a folder for each video file (30 in number)
            for sequence in range(0,self.no_sequences):
                self.create_dir(os.path.join(self.DATA_PATH, action, str(sequence)))
                
    
    def convert_to_1d(self, res) :
        a = []
        for i in res :
            for j in i :
                a.extend(j)
        a = np.array(a)
        return a
    
    
    def load_keypoints(self) :
        sequences, labels = [], []
        for action in self.actions:
            for sequence in np.array(os.listdir(os.path.join(self.DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    res = self.convert_to_1d(res)
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[action])
        return sequences, labels