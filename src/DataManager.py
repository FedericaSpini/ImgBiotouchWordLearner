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
            print("VANNO LETTI I PICKLE? Direi di sì....quando famo sto dataset poi lo mettemo a pickle")
        else:
            self.generate_data()
            # self.save_data()

        # self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0] è IL np.array che rappresenta un'immagine di BLOCK_LETTERS
        # della prima sessione del primo utente
        # print(self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0],
        #       type(self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0]))
        #
        # print(self.dataset.getData()['BLOCK LETTERS'][0].getImages()[0][0].shape)
    def save_data(self):
        dataset_pickle_path = ""
        pickle.dump(self.dataset, open(dataset_pickle_path, "wb"))

    def generate_data(self):
        self.dataset = PyImgDataset()
        # print(os.listdir("./test/"+self.dataset_name))
        dataset_path = Constants.DATASET_DIRECTORY_PATH+self.dataset_name+"/"
        for user_folder in os.listdir(dataset_path):
            # print(user_folder)
            user_folder_path = dataset_path+user_folder+"/"
            for session_number in os.listdir(user_folder_path):
                # print("    "+session_number)
                for writing_style in os.listdir(user_folder_path+session_number):
                    # print(writing_style)
                    currentSession = PyImgSession(user_folder, session_number) #I DATI DELLA SESSIONE
                    # print(s)
                    for img in os.listdir(user_folder_path+session_number+"/"+writing_style):
                        img_path = user_folder_path+session_number+"/"+writing_style+"/"+img
                        img_png= np.array(Image.open(img_path))
                        currentSession.add_image([np.array(img_png), ("s"+session_number+"."+img).replace(".png", "")])
                        # print(("s"+session_number+"."+img).replace(".png", ""))
                    self.dataset.add_session(writing_style, currentSession)


    @staticmethod
    def _check_saved_pickles(dataset_name):
        # for label in DATAFRAMES:
        #     if not Utils.os.path.isfile(Utils.BUILD_DATAFRAME_PICKLE_PATH(dataset_name, label)):
        #         return False
        # return True
        return False

if __name__=='__main__':
    print((os.path.dirname(os.path.abspath(__file__))).replace("src","")+"res\\")
    # d = DataManager(Constants.MINI_DATASET)
