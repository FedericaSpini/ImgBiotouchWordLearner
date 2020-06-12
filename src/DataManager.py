from src import Constants
from PIL import Image

import os
import numpy as np
import pickle

from src.PyImgDataset import PyImgDataset, PyImgSession


class DataManager:
    def __init__(self, dataset_name, update_data = False):
        self.dataset_name = dataset_name
        self.update_data = update_data
        self.images = []                    #????
        self.dataset = None                 #l'obiettivo è mettere il dataset fatto ad oggetti qui!
        self.load_images_data(self.update_data)


    def load_images_data(self, update_data = False):
        if not update_data and DataManager._check_saved_pickles(self.dataset_name):
            self.read_pickle_data()
        else:
            self.generate_data()
            self.save_pickle_data()
        # self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0] è IL np.array che rappresenta un'immagine di BLOCK_LETTERS
        # della prima sessione del primo utente
        # print(self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0],
        #       type(self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0]))
        #
        # print(self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0].shape)


    def read_pickle_data(self):
        dataset_pickle_path = Constants.PICKLE_DATA_DIRECTORY_PATH+self.dataset_name
        pickle_in = open(dataset_pickle_path, "rb")
        self.dataset = pickle.load(pickle_in)
        # print(self.dataset.getData())
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
                    currentSession = PyImgSession(user_folder, session_number) #I DATI DELLA SESSIONE
                    for img in os.listdir(user_folder_path+session_number+"/"+writing_style):
                        img_png= np.array(Image.open(user_folder_path+session_number+"/"+writing_style+"/"+img))
                        img_text_components = img.split(".")
                        img_text_components.remove('png')
                        # img_text_components IS SOMETHING LIKE: ['s0', 'u2', 'ITALIC', '28'] so [ session_code, class_code, writing_style, token_number_of_the_session]
                        img_text_components.insert(0,"s"+session_number)
                        currentSession.add_image([np.array(img_png), img_text_components])
                    self.dataset.add_session(writing_style, currentSession)

if __name__=='__main__':
    # for user_folder in os.listdir(Constants.DATASET_DIRECTORY_PATH):
    #     print(user_folder)
    # print(Constants.DATASET_DIRECTORY_PATH)
    d = DataManager(Constants.MINI_DATASET)
