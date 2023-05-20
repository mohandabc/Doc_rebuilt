from keras.callbacks import EarlyStopping, ModelCheckpoint
from distutils.command.config import config
import os
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, concatenate, Input
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

TRAIN_CONFIG = {
    'batch_size' : 64,
    'epochs' : 10,
}

def model_B():
    im_shapes = [(31, 31, 1), (63, 63, 1)]
    kernel_sizes = [4, 6]
    pool_sizes = [2, 4]
    n_filters = 60


    input_1 = Input(shape=im_shapes[0], name=f"level_1")
    W1 = Conv2D(filters = n_filters, kernel_size = kernel_sizes[0], activation='relu')(input_1)
    W1 = MaxPooling2D(pool_size = pool_sizes[0])(W1)

    input_2 = Input(shape=im_shapes[1], name=f"level_2")
    W2 = Conv2D(filters = n_filters, kernel_size = kernel_sizes[1], activation='relu')(input_2)
    W2 = MaxPooling2D(pool_size = pool_sizes[1])(W2)
  
    M = [W1, W2]
    M = concatenate(inputs = M)

    M = Flatten()(M)
    M = Dense(2, activation='softmax', name = "classification")(M)

    return Model(inputs = [input_1, input_2], outputs = M)

def model_RGB():
    return model_B() #tmp

class CNN():
    def __init__(self, model = "default"):
        self.config = TRAIN_CONFIG
        self._build_CNN(model)

    def set_batch_size(self, value):
        self.config['batch_size'] = value

    def set_epochs(self, value):
        self.config['epochs'] = value

    def _build_CNN(self, model):
        models = {
            'default':model_B,
            'RGB':model_RGB
        }
        self.model = models[model]()



    def train(self, train_gen, validation_gen, save_dir="./models"):
        callback = EarlyStopping(monitor='loss', patience=3)
        checkpoint_path = save_dir + "/model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_freq='epoch'
        )
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

        history = self.model.fit(train_gen,
            validation_data=validation_gen,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'], 
            callbacks=[callback, checkpoint],
            verbose=1,
        )
        return history


    def save(self, name):
        try:
            os.mkdir('model')
        except:
            pass
        self.model.save(f"model\\{name}.h5")

    def display(self, name="", graph=False):
        self.model.summary()
        if graph:
            try:
                plot_model(self.model, to_file=f"model\\{name}.png", show_shapes=True)
            except:
                pass 