
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict

from model.simple_autoencoder import SimpelAutoencoder
from model_exec.config import Config

def main():
    train_x, train_y = PreProcessing().make_train_data(Config.batch_size)
    encoder, autoencoder = SimpelAutoencoder.make_model()
    hist = Learning.run(autoencoder, train_x, train_y)

    SimpelAutoencoder.save_model(encoder ,"encode.hdf5")
    SimpelAutoencoder.save_model(autoencoder ,"autoencoder.hdf5")

if __name__ == '__main__':
   main()
