import os

import tensorflow as tf


NUM_CLASSES = 8
DATAPATH = '/data/05_inputs/'


def write_to_log(filepath, filename='log_file.txt'):
    name = f'{filename}'
    tf.io.write_file(os.path.join(DATAPATH, name), filepath)


def decode_image(image_bytes, channels=3):
    image = tf.io.decode_jpeg(image_bytes, channels=channels)
    return tf.cast(image, tf.float32) / 255.0


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

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "filepath": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    image_resized = decode_image(example["image"])
    label = decode_label(example['label'])

    return image_resized, label


def get_dataset(filenames, batch_size, repeat=1, autotune=tf.data.experimental.AUTOTUNE):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=autotune)
        .map(read_record, num_parallel_calls=autotune)
        .batch(batch_size, drop_remainder=True)
        .prefetch(autotune)
        .repeat(repeat)
    )
    return dataset
