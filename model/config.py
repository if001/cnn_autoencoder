from keras.optimizers import RMSprop, Adam
import os


class Config():
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_model = run_dir_path + "/weight/model.hdf5"
    loss = 'binary_crossentropy'
    loss = 'mean_squared_error'
    optimizer = 'adadelta'
    optimizer = 'adam'
    optimizer = Adam(lr=2e-4, beta_1=0.5)
    metrics = 'accuracy'
