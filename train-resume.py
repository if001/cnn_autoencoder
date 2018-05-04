from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning

from model_exec.config import Config
from model.simple_autoencoder import SimpleAutoencoder

def main():
    train_x, train_y = PreProcessing().make_train_data(Config.batch_size)
    autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")

    for i in range(10000):
        print("step: ",i)
        train_x, train_y = PreProcessing().make_train_data(Config.batch_size)
        hist = Learning.run(autoencoder, train_x, train_y)
        if i % 50 == 0:
            SimpleAutoencoder.save_model(autoencoder ,"autoencoder.hdf5")


if __name__ == '__main__':
   main()
