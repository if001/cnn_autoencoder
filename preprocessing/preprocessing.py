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


class PreProcessing(abc_preprocessing.ABCPreProcessing):
    @classmethod
    def make_train_data(cls, batch_size):
        image_list = []
        file_dir = config.Config.image_dir_path
        for file in os.listdir(file_dir):
            if (file.split(".")[-1] == "png") :
                filepath = file_dir + "/" + file
                print(filepath)
                img = Image.open(filepath)
                img = img.convert("RGB")
                img = img.resize((28, 28))
                img = np.array(img)
                image_list.append(img / 255.)

        image_list = np.array(image_list)
        print(image_list.shape)
        return image_list, image_list

    @classmethod
    def make_test_data(cls):
        pass

def main():
    pass


if __name__ == '__main__':
   main()
