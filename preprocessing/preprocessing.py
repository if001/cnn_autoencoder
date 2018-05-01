# from abc_preprocessing import ABCPreProcessing
from . import abc_preprocessing
from . import config

# import sys
# sys.path.append("../")
# from lib.data_shaping import DataShaping
import keras
from keras.datasets    import mnist
from PIL import Image


num_classes = 10

class PreProcessing(abc_preprocessing.ABCPreProcessing):
    @classmethod
    def make_train_data(cls):
        image_list = []
        file_dir = config.Config.image_dir_path
        for file in os.listdir(file_dir):
            filepath = file_dir + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            image = image.transpose(2, 0, 1)
            # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)

        return image_list, image_list

    @classmethod
    def make_test_data(cls):
        pass

def main():
    pass


if __name__ == '__main__':
   main()
