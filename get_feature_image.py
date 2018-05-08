from preprocessing.preprocessing import PreProcessing
from model_exec.predict import Predict

from model_exec.config import Config
from model.simple_autoencoder import SimpleAutoencoder

import numpy as np
import math
from PIL import Image
import random as rand


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_min_integer_sqrt(num):
    sq = math.sqrt(num)
    if (sq.is_integer()):
        return int(sq)
    else:
        return int(sq) + 1


def concat_img(img_list):
    # nimg_one_size = get_min_integer_sqrt(len(img_list))

    img_height = img_list[0].height
    img_width = img_list[0].height

    img = Image.new('RGB', (len(img_list) * img_width, img_height))

    for i in range(0, len(img_list) - 1):
        print(img_width * i, img_height)
        img.paste(img_list[i], (i * img_width, img_height))
    return img


def main():
    image_label_list = PreProcessing().make_feature_data()
    autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")

    encoder = SimpleAutoencoder.make_encoder_model(autoencoder)

    for _ in range(2):
        rand_num = rand.randint(0, len(image_label_list) - 1)
        image_label = image_label_list[rand_num]
        label, image = image_label

        score = Predict.run(encoder, np.array([image]))

        predict = np.array(score[0])
        predict.resize(8, 4, 4)

        img_list = []
        for value in predict:
            print(np.uint8(np.asarray(value)))
            predict_img = Image.fromarray(np.uint8(np.asarray(value)))
            img_list.append(predict_img.resize((4, 4), resample=0))

        print("label: ", label)
        img = concat_img(img_list)
        img.save('./save_feat_' + label + '.png')


if __name__ == '__main__':
    main()
