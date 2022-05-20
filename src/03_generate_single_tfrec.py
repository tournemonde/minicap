import pandas as pd
import tensorflow as tf
import argparse
from os import makedirs
from os.path import basename, join
from glob import glob
from tqdm import tqdm
from time import time
from yaml import safe_load

from sklearn.model_selection import train_test_split


def preprocess_image(image, image_wh):
    image = tf.reshape(image, [image.shape[0], -1])
    image = channelize(image)
    image = tf.image.resize(image, [*image_wh])
    return tf.cast(image, tf.uint8)


def channelize(x):
    return tf.stack([x, x, x], axis=-1)


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, label, filepath):
    feature = {
        "image": image_feature(image),
        "label": int64_feature(label),
        "filepath": bytes_feature(filepath),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "filepath": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=1)
    return example


def data_inputs(directory, eval=False, verbose=False, extension='jpeg'):
    images_list = glob(join(directory, '**/*.' + extension), recursive=True)

    label_pos = [(basename(i).rfind('_'), basename(i).rfind('.')) for i in images_list]

    # data_info = [(name, basename(name)[i + 1:j]) for i, j, name in zip(last_us, last_dots, images_list)]
    data_info = [{'filepath': name,
                  'file_id': basename(name)[:pos[1]],
                  'label': int(basename(name)[pos[0]+1:pos[1]])} for pos, name in zip(label_pos, images_list)]

    if not eval:
        train_val, test = train_test_split(data_info, test_size=0.1, random_state=71)
        train, val = train_test_split(train_val, test_size=0.1, random_state=71)

        data_dict = {'train': train, 'val': val, 'test': test}
        sizes = {'train': len(train), 'val': len(val), 'test': len(test)}

        if verbose:
            all_classes = [basename(name)[pos[0] + 1:pos[1]] for pos, name in zip(label_pos, images_list)]
            print(f'Classes: {sorted(list(set(all_classes)))}')
            print(f'train size: {sizes["train"]} - val size: {sizes["val"]} - test size: {sizes["test"]}')
            data_stats = [(basename(name)[:pos[1]], int(basename(name)[pos[0] + 1:pos[1]])) for pos, name in
                          zip(label_pos, images_list)]
            stats_df = pd.DataFrame(data_stats)
            print(stats_df.iloc[:, 1].value_counts().sort_values(ascending=False))

    else:
        data_dict = {'eval': data_info}
        sizes = {'eval': len(data_info)}

    return data_dict, sizes


def main(args):
    start = time()
    inputs_raw, input_sizes = data_inputs(args.data_dir, args.eval, verbose=True)
    
    params = safe_load(open(debug_prefix + "params.yaml"))["prepare"]
    img_wh = [params['img_width'], params['img_height']]

    processed_files_nr = 0
    for key in inputs_raw:
        output_dir = join(args.output_path, key)
        makedirs(output_dir, exist_ok=True)
        processed_files_nr += input_sizes[key]

        for rec_meta in tqdm(inputs_raw[key]):
            tf_record_name = join(output_dir, rec_meta['file_id'] + '.tfrec')
            with tf.io.TFRecordWriter(tf_record_name) as tf_writer:
                image = tf.io.decode_jpeg(tf.io.read_file(rec_meta['filepath']))
                image = preprocess_image(image, img_wh)
                example = create_example(image, rec_meta['label'], rec_meta['filepath'])
                tf_writer.write(example.SerializeToString())

    print(f'Process took {round(time() - start, 4)} seconds\nAn average of {round((time()-start)/processed_files_nr,4)} by image')


if __name__ == '__main__':
    debug_prefix = ''
    parser = argparse.ArgumentParser(description='Creates serialized data records')

    parser.add_argument('--data_dir', type=str, default= debug_prefix + 'data/03_primary/three_cat/', help='Folder with audio files to process')
    parser.add_argument('--output_path', type=str, default=debug_prefix + 'data/05_inputs/three_cat/', help='images output folder')
    parser.add_argument('--extension', type=str, default='jpeg', help='extension of the image files')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    parser.set_defaults(eval=False)

    args = parser.parse_args()

    main(args)
