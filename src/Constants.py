import os

IMG_DATASET = "anonymous_img_biotouch_2018_dataset"
IMG_MINI_DATASET = "mini_anonymous_img_biotouch_2018_dataset"
IMG_MEDIUM_DATASET = "medium_anonymous_img_biotouch_2018_dataset"


DATASET_DIRECTORY_PATH = 'D:/test/raccolta dei vari dataset/'
PICKLE_DATA_DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__)).replace("src","")+"res\\"
FOR_TORCH_FOLDER_SUFFIX = '_for_torch'

BLOCK_LETTERS = 'BLOCK_LETTERS'
ITALIC = 'ITALIC'