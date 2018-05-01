
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict

from model.simple_autoencoder import SimpelAutoencoder
from model.config import Config


def main():
    train_x, train_y = PreProcessing().make_train_data()

    encoder, autoencoder = SimpelAutoencoder.make_model()
    hist = Learning.run(autoencoder, train_x, train_y)

    SimpelAutoencoder.save_model(encoder ,save_fname)
    SimpelAutoencoder.save_model(autoencoder ,save_fname)



if __name__ == '__main__':
   main()
