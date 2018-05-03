from preprocessing.preprocessing import PreProcessing
from model_exec.predict import Predict

from model_exec.config import Config
from model.simple_autoencoder import SimpleAutoencoder

import numpy as np

from PIL import Image

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def main():
    image_label_list = PreProcessing().make_feature_data()
    autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")

    image_label = image_label_list[3]
    label, image = image_label

    score = Predict.run(autoencoder, np.array([image]))
    predict_img = np.array(score[0])*255
    predict_img = Image.fromarray(np.uint8(predict_img))
    print(label)
    def_img = Image.open("../string2image/image/" + label + ".png")
    def_img = def_img.convert("RGB")
    def_img = np.array(def_img)
    def_img = Image.fromarray(np.uint8(def_img))


    img = get_concat_h(def_img, predict_img)
    img.save('./save.png')
if __name__ == '__main__':
   main()
