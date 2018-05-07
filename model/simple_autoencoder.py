from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from . import abc_model
from . import config


class SimpleAutoencoder(abc_model.ABCModel):
    @classmethod
    def make_model(cls):
        input_img = Input(shape=(28, 28, 3))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        # encoder = Model(input_img, encoded)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer=config.Config.optimizer,
                            loss=config.Config.loss)
        autoencoder.summary()
        return autoencoder

    @classmethod
    def make_encoder_model(cls, autoencoder):
        encode = autoencoder.layers[0:7]

        input_img = Input(shape=(28, 28, 3))
        x = encode[1](input_img)
        x = encode[2](x)
        x = encode[3](x)
        x = encode[4](x)
        x = encode[5](x)
        encoded = encode[6](x)

        encoder = Model(input_img, encoded)
        encoder.summary()
        return encoder

    @classmethod
    def save_model(cls, model, fname):
        print("save" + config.Config.run_dir_path + "/weight/" + fname)
        model.save(config.Config.run_dir_path + "/weight/" + fname)

    @classmethod
    def load_model(cls, fname):
        print("load " + config.Config.run_dir_path + "/weight/" + fname)
        from keras.models import load_model
        return load_model(config.Config.run_dir_path + "/weight/" + fname)
