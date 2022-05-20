# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets.cifar100 import load_data
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.data.experimental import AUTOTUNE
import app_utils as app





NUM_CLASSES = 100
batch_size = 16
LEARNING_RATE = 0.001
EPOCHS = 1
# CLASS_WEIGHTS = params['class_weights']
#CLASS_WEIGHTS = {id: val for id, val in enumerate(CLASS_WEIGHTS)}
# DATAPATH = 'data/05_inputs/three_cat'
MODEL_OUTPUTS = 'data/02_models'
HISTORY_OUTPUT = 'data/03_meta'
MODEL_NAME = f'resnet_{LEARNING_RATE}_{batch_size}_{EPOCHS}.h5'



# tf.keras.datasets.cifar100.load_data(label_mode="fine")
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    print('done')