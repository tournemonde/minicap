import os
import tensorflow as tf
import matplotlib.pyplot as plt
from yaml import safe_load
from tensorflow.keras.applications.resnet50 import preprocess_input

# from utils.loader import DATAPATH

debug_prefix = ''

params = safe_load(open(debug_prefix + "params.yaml"))["train"]
AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_CLASSES = params['num_classes']
DATAPATH = 'data/05_inputs/'


def write_to_log(filepath, filename='log_file.txt'):
    name = f'{filename}'
    tf.io.write_file(os.path.join(DATAPATH, name), filepath)


def decode_image(image_bytes, filepath, channels=3):
    image = tf.io.decode_jpeg(image_bytes, channels=channels)
    return tf.cast(image, tf.float32) / 255.0


def decode_image_resnet(image_bytes, channels=3):
    image = tf.io.decode_jpeg(image_bytes, channels=channels)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return tf.cast(image, tf.float32) / 255.0


def decode_image_vgg19(image_bytes, channels=3):
    image = tf.io.decode_jpeg(image_bytes, channels=channels)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return tf.cast(image, tf.float32) / 255.0


def decode_image_mnet(image_bytes, channels=3):
    image = tf.io.decode_jpeg(image_bytes, channels=channels)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    return image # tf.cast(image, tf.float32) / 255.0


def decode_label(label):
    label = tf.cast(label, tf.int32)
    return tf.one_hot(label, NUM_CLASSES)


def read_record_simple(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "filepath": tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example, features)


def channelize(x):
    return tf.stack([x, x, x], axis=-1)


def read_record(example):
    try:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "filepath": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, features)
        #write_to_log(example['image'], filename='log2.txt')
        image_resized = decode_image(example["image"], example['filepath'])
        label = decode_label(example['label'])

        return image_resized, label
    except:
        print(example['filepath'])


def read_record_resnet(example):
    try:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, features)
        #write_to_log(example['image'], filename='log2.txt')
        image_resized = decode_image_resnet(example["image"])
        label = decode_label(example['label'])

        return image_resized, label
    except:
        print(example['filepath'])


def read_record_vgg19(example):
    try:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, features)
        image_resized = decode_image_vgg19(example["image"])
        label = decode_label(example['label'])

        return image_resized, label
    except:
        print(example['filepath'])


def read_record_mnet(example):
    # write_to_log(example['filepath'], filename='log.txt')
    try:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, features)
        image_resized = decode_image_mnet(example["image"])
        label = decode_label(example['label'])

        return image_resized, label
    except:
        print(example['filepath'])


def get_dataset(filenames, batch_size, repeat=1):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(read_record, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
        .repeat(repeat)
    )
    return dataset


def get_dataset_vgg19(filenames, batch_size, repeat=1):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(read_record_vgg19, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
        .repeat(repeat)
    )
    return dataset


def get_dataset_resnet(filenames, batch_size, repeat=1):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(read_record_resnet, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
        .repeat(repeat)
    )
    return dataset


def get_dataset_mnet(filenames, batch_size, repeat=1):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(read_record_mnet, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
        .repeat(repeat)
    )
    return dataset
