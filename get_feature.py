from preprocessing.preprocessing import PreProcessing
from model_exec.predict import Predict

from model_exec.config import Config
from model.simple_autoencoder import SimpleAutoencoder

import numpy as np


def main():
    image_label_list = PreProcessing().make_feature_data()
    autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")

    encoder = SimpleAutoencoder.make_encoder_model(autoencoder)

    for image_label in image_label_list:
        label, image = image_label
        score = Predict.run(encoder, np.array([image]))
        print(label, score)


if __name__ == '__main__':
   main()
