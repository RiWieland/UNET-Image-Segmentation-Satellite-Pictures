
import os

import string
import random

def extract_ids(str_):
    '''
    Function extracts ID from Filename
    '''
    # remove .jpg
    str_ = str_[:-4]

    # remove zeros
    for counter, i in enumerate(str_):
        if i =='0':
            continue
        else:
            str_ = str_[int(counter):]
            break

    return int(str_)


def get_dir_dict():
    '''
    The Directory Dictionary
    '''

    dict_dir = {'Train':
                {
                    'Ann_Path': os.getcwd() + "/Data/raw/Annotations/Train/",
                    'Ann_Small': os.getcwd() + "/Data/raw/Annotations/Train/annotation.json",
                    'Image_Path': os.getcwd() + "/Data/raw/Image/Train/Images/",
                    'Mask_Path': os.getcwd() + "/Data/Mask/Train/"

                },
            'Val':
                {
                    'Ann_Path': os.getcwd() + "/Data/raw/Annotations/Val/",
                    'Ann_Small': os.getcwd() + "/Data/raw/Annotations/Val/annotation.json",
                    'Image_Path': os.getcwd() + "/Data/raw/Image/Val/Images/",
                    'Mask_Path': os.getcwd() + "/Data/Mask/Val/"
                },

            'Test':
                {
                    'Test_Path': os.getcwd() + "/Data/raw/Image/Test/"
                },
            'Log':
                {
                    'Log_path': os.getcwd() + "/Log/"
                },
            'Saved_Models':
                {
                    'Save_Path': os.getcwd() + "/Saved_Models/"
                },
            'Predictions':
                {
                    'Predicitons_Path': os.getcwd() + '/Data/Predictions/'
                }
            }
    return dict_dir


def create_dir(dir_dict):

    for key, value in dir_dict.items():
        for path in dir_dict[key].values():
            try:
                if not os.path.exists(path):
                    os.makedirs(path)
                    print("Directory ", path, " created!")
            except:
                continue


def id_generator(size=6, chars=string.digits):
    return ''.join(random.choice(chars) for x in range(size))


