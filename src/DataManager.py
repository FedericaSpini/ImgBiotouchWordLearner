import shutil

import Constants
from PIL import Image
from shutil import copyfile

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
            self.create_simple_dataset_folder()
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
        torch_data_path = Constants.DATASET_DIRECTORY_PATH+self.dataset_name+Constants.FOR_TORCH_FOLDER_SUFFIX
        self.dataset = PyImgDataset()
        dataset_path = Constants.DATASET_DIRECTORY_PATH+self.dataset_name+"/"
        for user_folder in os.listdir(dataset_path):
            user_folder_path = dataset_path+user_folder+"/"
            for session_number in os.listdir(user_folder_path):
                for writing_style in os.listdir(user_folder_path+session_number):
                    if not os.path.exists(torch_data_path + '/'+writing_style):
                        os.mkdir(torch_data_path + '/'+writing_style)
                        print("Directory ", torch_data_path + '/'+writing_style, " Created ")
                    if not os.path.exists(torch_data_path + '/'+writing_style+'/'+user_folder):
                        os.mkdir(torch_data_path + '/'+writing_style+'/'+user_folder)
                        print("Directory ", torch_data_path + '/'+writing_style+'/'+user_folder, " Created ")
                    currentSession = PyImgSession(user_folder, session_number, writing_style) #The data of the single recording session, of a given user in a give writing_style
                    for img in os.listdir(user_folder_path+session_number+"/"+writing_style):
                        img_png= np.array(Image.open(user_folder_path+session_number+"/"+writing_style+"/"+img))
                        copyfile(user_folder_path+session_number+"/"+writing_style+'/'+img, torch_data_path + '/'+writing_style+'/'+user_folder+ '/'+img)
                        # print(img)
                        currentSession.add_image(np.array(img_png))
                    self.dataset.add_session(writing_style, currentSession)

    def create_simple_dataset_folder(self):
        dirName = Constants.DATASET_DIRECTORY_PATH+self.dataset_name+Constants.FOR_TORCH_FOLDER_SUFFIX
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory ", dirName, " Created ")
        else:
            for filename in os.listdir(dirName):
                file_path = os.path.join(dirName, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            print("Directory ", dirName, " already exists")


if __name__=='__main__':
    # for user_folder in os.listdir(Constants.DATASET_DIRECTORY_PATH):
    #     print(user_folder)
    # print(Constants.DATASET_DIRECTORY_PATH)
    d = DataManager(Constants.MINI_DATASET)
