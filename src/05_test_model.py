import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from os import makedirs
from os.path import join
from utils.loader import *
from yaml import safe_load

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from utils.train_utils import get_dataset_mnet, get_dataset_vgg19
from time import time


debug_prefix = ''
params = safe_load(open(debug_prefix + "params.yaml"))["train"]
NUM_CLASSES = params['num_classes']
batch_size = params['batch_size']
LEARNING_RATE = params['learning_rate']
EPOCHS = params['epochs']


def granular_evaluation(filenames, model, batch_size, architecture='mnet', eval=False):
    if architecture == 'mnet':
        dataset = get_dataset_mnet(filenames, batch_size=batch_size)
    elif architecture == 'vgg':
        dataset = get_dataset_vgg19(filenames, batch_size=batch_size)
    image_dataset = dataset.map(lambda image, label: image)
    probs = model.predict(image_dataset)
    y_preds = np.argmax(probs, axis=-1)

    if not eval:
        label_dataset = dataset.map(lambda image, label: label).unbatch()
        labels_oh = next(iter(label_dataset.batch(len(filenames)))).numpy()
        return np.argmax(labels_oh, axis=-1), y_preds
    else:
        return None, y_preds


# TODO nicer and annotate
def plot_cm(cm, classes: list):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.matshow(cm)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontdict={'fontsize': 6})
    plt.setp(ax.get_xticklabels(), rotation=65, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontdict={'fontsize': 6})
    plt.setp(ax.get_yticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    plt.savefig(join(args.report_folder, args.model + '_cm.png'))


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


def main(args):

    start = time()

    if not args.eval:
        report_folder = args.report_folder
        makedirs(report_folder, exist_ok=True)

    print(f'model name: {join(args.model_path, args.model)} \nbatch size: {batch_size}')
    model = load_model(join(args.model_path, args.model))

    if args.eval:
        print(f'{args.in_dir}*.tfrec')
        file_dict = {'eval': tf.io.gfile.glob(f'{args.in_dir}*.tfrec')}
    else:
        dataset_types = ['train', 'val', 'test']
        file_dict = {dst: tf.io.gfile.glob(f'{join(args.in_dir, dst)}*.tfrec') for dst in dataset_types}

    processed_files_nr = 0
    for dst in file_dict:
        print('\n' + '=' * 25 + f' <<<{dst}>>> ' + "=" * 25 + '\n')
        processed_files_nr += len(file_dict[dst])

        y_labels, y_preds = granular_evaluation(file_dict[dst], model, batch_size, architecture=args.architecture, eval=args.eval)
        if not args.eval:
            evaluation_metrics(y_labels, y_preds, list(range(NUM_CLASSES)))
        else:
            unique, counts = np.unique(y_preds, return_counts=True)
            count_dict = dict(zip(unique, counts))
            print(count_dict)

    print(f'Process took {round(time() - start, 4)} seconds\nAn average of {round((time() - start) / processed_files_nr, 4)} by file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process sound files and attach labels.')

    parser.add_argument('--in_dir', type=str, default=debug_prefix + 'data/05_inputs/check/', help='Folder with serialized files')
    parser.add_argument('--model_path', type=str, default=debug_prefix + 'data/06_models_tuned/', help='path of the model')
    parser.add_argument('--model', type=str, help='name of the model to be tested')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    parser.set_defaults(eval=True)
    parser.add_argument('--report_folder', default='data/07_report', type=str, help='name of the model to be tested')
    parser.add_argument('--architecture', default='mnet', type=str, help='architecture of the model')

    args = parser.parse_args()
    # args.model = '3class_VC_mnetv2_step2_3_1e-05_64_111.h5'
    # args.eval = False

    main(args)




