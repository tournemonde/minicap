import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from os import makedirs
from os.path import join
from yaml import safe_load

from tensorflow.keras.metrics import Precision, Recall

from utils.loader import *
from utils.train_utils import *
from utils.models import build_mnetv2

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score

from clearml import Task


task = Task.init(project_name='testProject', task_name='base model: three cats - TEST')
debug_prefix = ''
params = safe_load(open(debug_prefix + "params.yaml"))["train"]

NUM_CLASSES = params['num_classes']
batch_size = params['batch_size']
LEARNING_RATE = params['learning_rate']
EPOCHS = params['epochs']
CLASS_WEIGHTS = params['class_weights']
CLASS_WEIGHTS = {id: val for id, val in enumerate(CLASS_WEIGHTS)}


AUTOTUNE = tf.data.experimental.AUTOTUNE
DATAPATH = 'data/05_inputs/three_cat'
MODEL_OUTPUTS = 'data/06_models'
HISTORY_OUTPUT = 'data/08_meta'
MODEL_NAME = f'3class_VC_mnetv2_step2_{NUM_CLASSES}_{LEARNING_RATE}_{batch_size}_{EPOCHS}.h5'
TENSORBOARD_LOG_DIR = "logs/tb_logs"
REPORT_FOLDER = 'data/07_report'
makedirs(REPORT_FOLDER, exist_ok=True)

reload_model = True
reloaded_model = '3class_VC_mnetv2_step1_3_0.0001_64_150.h5' 


def plot_hist(hist):
    plt.clf()
    plt.plot(hist.history["precision"])
    plt.plot(hist.history["val_precision"])
    plt.title("model precision")
    plt.ylabel("precision")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

    makedirs('data/07_report/', exist_ok=True)
    plt.savefig(f"data/07_report/{MODEL_NAME}_history.png")


def plot_fit_hist(hist):
    plt.clf()
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model fit")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

    makedirs('data/07_report/', exist_ok=True)
    plt.savefig(f"data/07_report/{MODEL_NAME}_fit_history.png")


def granular_evaluation(filenames, batch_size, model):
    dataset = get_dataset_mnet(filenames, batch_size=batch_size)
    image_dataset = dataset.map(lambda image, label: image)
    label_dataset = dataset.map(lambda image, label: label).unbatch()
    labels_oh = next(iter(label_dataset.batch(len(filenames)))).numpy()
    y_labels = np.argmax(labels_oh, axis=-1)

    probs = model.predict(image_dataset)
    y_preds = np.argmax(probs, axis=-1)
    return y_labels, y_preds


# TODO nicer and annotate
def plot_cm(cm, classes: list):
    plt.clf()
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.matshow(cm)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontdict={'fontsize': 6})
    plt.setp(ax.get_xticklabels(), rotation=65, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontdict={'fontsize': 6})
    plt.setp(ax.get_yticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    plt.savefig(join(REPORT_FOLDER, MODEL_NAME + '_cm.png'))


def evaluation_metrics(labels, preds, classes, plot=True):
    cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))
    print(cm)

    prec = precision_score(labels, preds, labels=range(NUM_CLASSES), average='macro', zero_division=0)
    rec = recall_score(labels, preds, labels=range(NUM_CLASSES), average='macro', zero_division=0)
    f1 = f1_score(labels, preds, labels=range(NUM_CLASSES), average='macro', zero_division=0)

    print('Precision \t: {:.4f}'.format(prec))
    print('Recall \t\t: {:.4f}'.format(rec))
    print('F1 score \t: {:.4f}'.format(f1))

    if plot:
        plot_cm(cm, classes)


def main():
    lr_schedule_cb = tf.keras.optimizers.schedules.ExponentialDecay(
         LEARNING_RATE, decay_steps=20, decay_rate=0.96, staircase=True
    )

    # callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        join(MODEL_OUTPUTS, MODEL_NAME), save_best_only=True
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience=25,
                                                         restore_best_weights=True
                                                         )

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1)

    # with strategy.scope():
    # model = build_vgg19_functional(NUM_CLASSES)

    model = build_mnetv2(NUM_CLASSES, fine_tune_layer=-23)

    if reload_model:
        model.load_weights(join(MODEL_OUTPUTS, reloaded_model))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  loss_weights= [0.01, .1, 1],
                  metrics=[Precision(name='precision'), Recall(name='recall')])

    print(model.summary())

    dataset_types = ['train', 'val', 'test']
    dst = dataset_types[0]
    train_filenames = tf.io.gfile.glob(f'{join(DATAPATH, dst)}/*.tfrec')
    dst = dataset_types[1]
    val_filenames = tf.io.gfile.glob(f'{join(DATAPATH, dst)}/*.tfrec')
    # train_filenames += val_filenames
    train_size = len(train_filenames)
    steps_per_epoch = train_size // batch_size
    dst = dataset_types[2]
    test_filenames = tf.io.gfile.glob(f'{join(DATAPATH, dst)}/*.tfrec')

    hist = model.fit(
        x=get_dataset_mnet(train_filenames, batch_size, EPOCHS),
        epochs=EPOCHS,
        validation_data=get_dataset_mnet(val_filenames, batch_size),
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1,
        class_weight=CLASS_WEIGHTS
    )

    plot_hist(hist)

    print('==== TRAIN ====')
    y_labels, y_preds = granular_evaluation(train_filenames, batch_size, model)
    evaluation_metrics(y_labels, y_preds, list(range(NUM_CLASSES)))

    print('==== VALIDATION ====')
    y_labels, y_preds = granular_evaluation(val_filenames, batch_size, model)
    evaluation_metrics(y_labels, y_preds, list(range(NUM_CLASSES)))

    print('==== TEST ====')
    y_labels, y_preds = granular_evaluation(test_filenames, batch_size, model)
    evaluation_metrics(y_labels, y_preds, list(range(NUM_CLASSES)))

    plot_fit_hist(hist)


if __name__ == '__main__':
    main()
    print('finished training')
