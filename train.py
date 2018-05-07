
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning

from model.simple_autoencoder import SimpleAutoencoder
from model_exec.config import Config

date_size = 40000
test_size = 10000



def main():
    autoencoder = SimpleAutoencoder.make_model()
    cbs = SimpleAutoencoder.set_callbacks("autoencoder.hdf5")

    train_x, train_y = PreProcessing().make_train_data(date_size)
    test_x, test_y = PreProcessing().make_train_data(test_size)
    hist = Learning.run_with_test(
        autoencoder, train_x, train_y, test_x, test_y, cbs)

    # SimpleAutoencoder.save_model(autoencoder, "autoencoder.hdf5")


if __name__ == '__main__':
    main()
