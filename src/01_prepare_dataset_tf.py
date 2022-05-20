import argparse
import json
import tensorflow_io as tfio
import tensorflow as tf

import yaml

import numpy as np
import pandas as pd

from PIL import Image
from math import floor

from os import makedirs
from os.path import join, dirname, basename
from pathlib import Path
from glob import glob
from collections import namedtuple

from tqdm import tqdm
from datetime import timedelta
from typing import Union
from time import time


LABELS_KEY_NAME = "tricks"
AudParams = namedtuple('AudParams', ['window_size', 'step_size', 'sample_rate', 'audio_start_ts'])
SpectParams = namedtuple('SpectParams', ['img_width', 'img_height'])
MEASUREMENT_SHIFT = 1.77
debug_prefix = ''

# file_list = ['2021-08-09T18-19-42.421451.audio.mp3', '2021-08-09T14-59-34.360217.audio.mp3',
#              '2021-08-09T19-20-44.766203.audio.mp3', '2021-08-09T13-20-30.854987.audio.mp3',
#              '2021-08-09T18-40-43.569015.audio.mp3', '2021-08-09T15-00-34.860809.audio.mp3',
#              '2021-08-09T19-19-44.333405.audio.mp3', '2021-08-09T13-19-30.265910.audio.mp3']

file_list = ['2021-08-09T12-40-29.587611.audio.mp3']


def timestamp_from_string(s: str, first_str_after_date: str ='.audio', format: str ='%Y-%m-%dT%H-%M-%S.%f'):
    return pd.to_datetime(s[:s.find(first_str_after_date)], format=format)


def load_audio_sample(file_name):
    """
    load_audio_sample loads an audio sample from any file (mp3, wav, flac, etc.)

    It returns the audio as a numpy array, normalized, together with its frame-rate
    """
    sample = tfio.audio.AudioIOTensor(filename=file_name)
    sample_fs = sample.rate.numpy()
    norm_audio_data = tf.cast(sample[:len(sample)], tf.float32) / 32768.0
    norm_audio_data = tf.reduce_mean(norm_audio_data, axis=1)

    audio_start_ts = timestamp_from_string(basename(file_name))

    return norm_audio_data, sample_fs, audio_start_ts


def divide_sample(audio_data, audio_params: AudParams):
    """
    Break signal into list of segments and returns them
    """
    signal_len = len(audio_data)
    segment_size = int(audio_params.window_size * audio_params.sample_rate)  # segment size in samples
    step_size = int(audio_params.step_size * audio_params.sample_rate) 
    
    segments = np.array([audio_data[x:x + segment_size] for x in
                        np.arange(0, signal_len, step_size)], dtype=object)
    return segments


def remove_silent_segments(segments, threshold):
    """
    Remove empty segments (without cars) using an energy threshold = 50% (by default) of the median energy
    """
    energies = [(s**2).sum() / len(s) for s in segments]
    # (attention: integer overflow would occure without normalization here!)
    thres = threshold * np.median(energies)
    index_of_segments_to_keep = (np.where(energies > thres)[0])
    # get segments that have energies higher than a the threshold:
    segments2 = segments[index_of_segments_to_keep] 

    return segments2


def structure_data(data: pd.DataFrame, mapping_dict, shift=None):

    data['ts'] = pd.to_datetime(data['Zeit NTP'], unit='s')
    if shift is not None:
        data['ts'] = data['ts'] + timedelta(seconds=shift)
    data['labels'] = data['Fahrzeugklasse'].map(mapping_dict)
    data = data.astype({"labels": 'category'})

    return data[['ts', 'labels', 'Spur', 'Geschwindigkeit']]


def get_labels_from_folder(directory: str, extension: str = 'csv'):
    """
        Read the labels for a specific audio file and return it as a dataframe
    """
    data = pd.DataFrame()
    for file_name in glob(join(directory, '*.' + extension)):
        x = pd.read_csv(file_name, low_memory=False)
        x['filename'] = basename(file_name)
        x['timestamp_from_name'] = timestamp_from_string(basename(file_name),
                                                         first_str_after_date='_Lfd',
                                                         format='%Y_%m_%d_%H_%M')

        data = pd.concat([data, x], axis=0)
    return data[data['Fahrzeugklasse'].notnull()]


def get_labels(label_file, file_name):
    """
    Read the labels for a specific audio file and return it as a dataframe 
    """
    with open(label_file, 'rt') as f:
        j = json.loads(f.read())

    df = None
    file_name = Path(file_name).with_suffix('').stem
    
    for i in j:
        if file_name in i['videoSource']:
            df = pd.read_json(json.dumps(i[LABELS_KEY_NAME]), orient='records')
            break
    
    if df is None:
        return None

    df['start_in_secs'] = df['start'] / 60
    df['duration'] = df['end'] - df['start']
    df['labels'] = df['labels'].map(lambda x: ', '.join(x))
    df = df.astype({"labels":'category'})

    return df


def scale_minmax_old(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def scale_minmax(X, min=tf.constant([0.0]), max=tf.constant([255.0])):
    min_x = tf.reduce_min(X)
    max_x = tf.reduce_max(X)
    x_std = (tf.subtract(X, min_x)) / (tf.subtract(max_x, min_x))
    return tf.add(tf.multiply(x_std, tf.subtract(max, min)), min)


def match_labels_to_seq(segments, df: Union[pd.DataFrame, None], audio_params: AudParams, spect_params: SpectParams, out_dir: str, train: bool = True):
    """
    The heart of the labeling:
    check if the the image contains a vehicle, convert to mel spectogram and save the image with the according label
    if df is None the function is in prediction modus. Images will not be class labeled
    """
    threshold_end = timedelta(seconds=audio_params.step_size / 10)
    threshold_start = threshold_end * 3

    for idx, seg in enumerate(segments):
        n_mels = spect_params.img_height # number of bins in spectrogram. Height of image
        hop_length = floor(audio_params.window_size * audio_params.sample_rate / spect_params.img_width)  # number of samples per time-step in spectrogram

        spectrogram = tfio.audio.spectrogram(seg, nfft=hop_length*4, window=224, stride=hop_length)
        mel_spectrogram = tfio.audio.melscale(spectrogram, rate=audio_params.sample_rate,
                                              mels=n_mels, fmin=0, fmax=audio_params.sample_rate/2)
        mel_db = tfio.audio.dbscale(mel_spectrogram, top_db=150).numpy()
        mels_db_t = tf.transpose(mel_db)
        img = tf.cast(scale_minmax(mels_db_t), tf.uint8)
        img = tf.experimental.numpy.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255 - img

        start = audio_params.audio_start_ts + timedelta(seconds=audio_params.step_size) * idx
        end = start + timedelta(seconds=audio_params.window_size)

        # adding a threshold to the vehicle search
        thresh_start = start - threshold_start
        thresh_end = end - threshold_end

        if df is not None:
            class_codes = [0]
            df_parts = df[((df["ts"] >= thresh_start) & (df["ts"] <= thresh_end))]
            if len(df_parts) > 0:
                class_codes = set(df_parts['labels'].to_list())
        else:
            class_codes = [99]

        im = Image.fromarray(img.numpy())
        for class_code in class_codes:
            filename = join(out_dir, f"img_{idx}_{start}_{end}_{class_code}.jpeg")
            makedirs(dirname(filename), exist_ok=True)
            im.save(filename)


def main(args):
    start = time()
    params = yaml.safe_load(open(debug_prefix + "params.yaml"))["prepare"]

    spect_params = SpectParams(img_width=params["img_width"], 
                               img_height=params['img_height'])

    if not args.eval:
        data = get_labels_from_folder(args.label_folder)
        data = structure_data(data, params['class_equivalence'], shift=MEASUREMENT_SHIFT)
    else:
        data = None

    for idx, audio_file in enumerate(tqdm(glob(join(args.in_dir, "**/*.mp3")))):
        # print(f" >>> Processing {audio_file} <<< ")
        if basename(audio_file) in file_list:

            audio_data, sample_fs, audio_start_ts = load_audio_sample(audio_file)
            audio_params = AudParams(window_size=params["window_size_in_sec"],
                                        step_size=params["step_size_in_sec"],
                                        sample_rate=sample_fs,
                                        audio_start_ts=audio_start_ts)

            segments = divide_sample(audio_data, audio_params)

            outdir = join(args.out_dir, str(audio_start_ts).replace(' ', '_'))
            makedirs(outdir, exist_ok=True)
        else:
            continue

        match_labels_to_seq(segments,
                            data,
                            audio_params,
                            spect_params,
                            outdir,
                            params["class_mapping"])

    print(f'Process took {round(time() - start, 4)} seconds\nAn average of {round((time()-start)/len(segments),4)} by segment')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sound files and attach labels.')

    parser.add_argument('--in_dir', type=str, default=debug_prefix + "data/01_raw/2021-08-09", help='Folder with audio files to process')
    parser.add_argument('--out_dir', type=str, default=debug_prefix + "data/03_primary/2021-08-09_tf/", help='images output folder')
    parser.add_argument('--label_folder', type=str, default=debug_prefix + "data/02_labels/srb2021-08-09_labels", help='Label file')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    parser.set_defaults(eval=False)

    args = parser.parse_args()
    # args.eval = True

    main(args)
