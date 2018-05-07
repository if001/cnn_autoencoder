from keras.optimizers import RMSprop
import os


class Config():
    # batch_size = 256
    # test_size = 128
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_model = run_dir_path + "/weight/model.hdf5"

    loss = 'categorical_crossentropy'
    optimizer = RMSprop()
    metrics = 'accuracy'
