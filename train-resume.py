from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning

from model_exec.config import Config
from model.simple_autoencoder import SimpleAutoencoder

import sys

date_size = 20000
test_size = 10000


def main():
    train_x, train_y = PreProcessing().make_train_data(Config.batch_size)
    cbs = SimpleAutoencoder.set_callbacks("autoencoder.hdf5")
    autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")
    
    train_x, train_y = PreProcessing().make_train_data(date_size)
    test_x, test_y = PreProcessing().make_train_data(test_size)
    hist = Learning.run_with_test(
        autoencoder, train_x, train_y, test_x, test_y, cbs)
    # SimpleAutoencoder.save_model(autoencoder, "autoencoder.hdf5")


if __name__ == '__main__':
    main()
