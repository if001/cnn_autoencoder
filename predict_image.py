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
        return sq
    else:
        return int(sq) + 1


def concat_img(img_list):
    img_one_size = get_min_integer_sqrt(len(img_list))

    img_height = img_list[0].height
    img_width = img_list[0].width

    img = Image.new(
        'RGB', (img_width * img_one_size, img_height * img_one_size))

    cnt = 0
    for i in range(0, img_one_size - 1):
        for j in range(0, img_one_size - 1):
            if len(img_list) > cnt:
                img.paste(img_list[cnt], (i * img_width, j * img_height))
            cnt += 1
    img.save('./save.png')


def main():
    image_label_list = PreProcessing().make_feature_data()
    autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")
    img_list = []

    for _ in range(300):
        rand_num = rand.randint(0, len(image_label_list) - 1)
        image_label = image_label_list[rand_num]
        label, image = image_label

        score = Predict.run(autoencoder, np.array([image]))
        predict_img = np.array(score[0]) * 255
        predict_img = Image.fromarray(np.uint8(predict_img))
        print("label: ", label)
        def_img = Image.open("../string2image/image/" + label + ".png")
        def_img = def_img.convert("RGB")
        def_img = np.array(def_img)
        def_img = Image.fromarray(np.uint8(def_img))
        img_list.append(get_concat_h(def_img, predict_img))

    concat_img(img_list)


if __name__ == '__main__':
    main()
