from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from . import abc_model
from . import config


class SimpelAutoencoder(abc_model.ABCModel):
    @classmethod
    def make_model(cls):
        input_img = Input(shape=(28, 28, 3))

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        encoder = Model(input_img, encoded)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.summary()
        return encoder, autoencoder

    @classmethod
    def save_model(cls, model, fname):
        print("save"+  config.Config.run_dir_path + "/" + fname)
        model.save(config.Config.run_dir_path + "/" + fname)

    @classmethod
    def load_model(cls):
        print("load "+  config.Config.run_dir_path + "/" + fname)
        from keras.models import load_model
        return load_model(config.Config.run_dir_path + "/" + fname)




