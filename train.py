
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning

from model.simple_autoencoder import SimpleAutoencoder
from model_exec.config import Config

def main():
    train_x, train_y = PreProcessing().make_train_data(Config.batch_size)
    autoencoder = SimpleAutoencoder.make_model()

    hist = Learning.run(autoencoder, train_x, train_y)

    SimpleAutoencoder.save_model(autoencoder ,"autoencoder.hdf5")

if __name__ == '__main__':
   main()
