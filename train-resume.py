from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning

from model_exec.config import Config
from model.simple_autoencoder import SimpleAutoencoder

import sys

def main():
    train_x, train_y = PreProcessing().make_train_data(Config.batch_size)
    autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")

    start = 0
    if "i==" in sys.argv[-1]:
        start = sys.argv[-1].split("==")[-1]

    for i in range(10000):
        print("step: ",i)
        train_x, train_y = PreProcessing().make_train_data(Config.batch_size)
        test_x, test_y = PreProcessing().make_train_data(Config.test_size)
        hist = Learning.run_with_test(autoencoder, train_x, train_y, test_x, test_y)
        if i % 50 == 0:
            SimpleAutoencoder.save_model(autoencoder ,"autoencoder.hdf5")


if __name__ == '__main__':
   main()
