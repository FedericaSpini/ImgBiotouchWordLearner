import Constants
from PIL import Image

import os
import numpy as np
import pickle

from PyImgDataset import PyImgDataset
from PyImgSession import PyImgSession


class DataManager:
    def __init__(self, dataset_name, update_data = False):
        self.dataset_name = dataset_name
        self.update_data = update_data
        self.dataset = None                 #l'obiettivo è mettere il dataset fatto ad oggetti qui!
        self.load_images_data(self.update_data)
        # print('----')
        # print(len(self.dataset.getData()['ITALIC'][0].getImages()))
        # print(False in ((self.dataset.getData()['ITALIC'][0].getImages()[0])==(self.dataset.getData()['ITALIC'][1].getImages()[0])))
        # print((self.dataset.getData()['ITALIC'][0].user_id)==(self.dataset.getData()['ITALIC'][1].user_id))


    def load_images_data(self, update_data = False):
        if not update_data and DataManager._check_saved_pickles(self.dataset_name):
            self.read_pickle_data()
        else:
            self.generate_data()
            self.save_pickle_data()


    def read_pickle_data(self):
        dataset_pickle_path = Constants.PICKLE_DATA_DIRECTORY_PATH+self.dataset_name
        pickle_in = open(dataset_pickle_path, "rb")
        self.dataset = pickle.load(pickle_in)
        pickle_in.close()


    def save_pickle_data(self):
        dataset_pickle_path = Constants.PICKLE_DATA_DIRECTORY_PATH+self.dataset_name
        pickle_out = open(dataset_pickle_path, "wb")
        pickle.dump(self.dataset, pickle_out)
        print("PICKLES FILE ARE SAVED!")
        pickle_out.close()

    @staticmethod
    def _check_saved_pickles(dataset_name):
        return os.path.isfile(Constants.PICKLE_DATA_DIRECTORY_PATH+dataset_name)

    def generate_data(self):
        self.dataset = PyImgDataset()
        dataset_path = Constants.DATASET_DIRECTORY_PATH+self.dataset_name+"/"
        for user_folder in os.listdir(dataset_path):
            user_folder_path = dataset_path+user_folder+"/"
            for session_number in os.listdir(user_folder_path):
                for writing_style in os.listdir(user_folder_path+session_number):
                    currentSession = PyImgSession(user_folder, session_number, writing_style) #The data of the single recording session, of a given user in a give writing_style
                    for img in os.listdir(user_folder_path+session_number+"/"+writing_style):
                        img_png= np.array(Image.open(user_folder_path+session_number+"/"+writing_style+"/"+img))
                        # print(img)
                        currentSession.add_image(np.array(img_png))
                    self.dataset.add_session(writing_style, currentSession)

if __name__=='__main__':
    # for user_folder in os.listdir(Constants.DATASET_DIRECTORY_PATH):
    #     print(user_folder)
    # print(Constants.DATASET_DIRECTORY_PATH)
    d = DataManager(Constants.MINI_DATASET)
