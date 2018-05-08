from keras.optimizers import RMSprop
import os


class Config():
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_model = run_dir_path + "/weight/model.hdf5"
    loss = 'mean_squared_error'
    loss = 'binary_crossentropy'
    optimizer = 'adadelta'
    optimizer = 'adam'
    metrics = 'accuracy'
