# from abc_preprocessing import ABCPreProcessing
from . import abc_preprocessing
from . import config

# import sys
# sys.path.append("../")
# from lib.data_shaping import DataShaping
import keras
from keras.datasets    import mnist
from PIL import Image
import numpy as np
import os
import random as rand

class PreProcessing(abc_preprocessing.ABCPreProcessing):
    @classmethod
    def make_train_data(cls, batch_size):
        image_list = []
        file_dir = config.Config.image_dir_path
        files = os.listdir(file_dir)

        while(True):
            idx = rand.randint(0, len(files)-1)
            if (files[idx].split(".")[-1] == "png") :
                filepath = file_dir + "/" + files[idx]
                img = Image.open(filepath)
                img = img.convert("RGB")
                img = img.resize((28, 28))
                img = np.array(img)
                image_list.append(img / 255.)
            if len(image_list) == batch_size: break

        image_list = np.array(image_list)
        return image_list, image_list

    @classmethod
    def make_feature_data(cls):
        image_label_list = []
        file_dir = config.Config.image_dir_path

        for fname in os.listdir(file_dir):
            if (fname.split(".")[-1] == "png") :
                filepath = file_dir + "/" + fname
                img = Image.open(filepath)
                img = img.convert("RGB")
                img = img.resize((28, 28))
                img = np.array(img)
                image_label_list.append([fname.split(".")[0], img / 255.])

        return image_label_list

    @classmethod
    def make_test_data(cls):
        pass

    
def main():
    pass


if __name__ == '__main__':
   main()
