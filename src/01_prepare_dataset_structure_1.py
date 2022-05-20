import argparse
import json
import yaml

import numpy as np
import pandas as pd

import librosa
import librosa.display

from PIL import Image
from math import floor

from functools import partial
from os import makedirs
from os.path import join, dirname
from pathlib import Path
from glob import glob
from collections import namedtuple

from pydub import AudioSegment
from tqdm import tqdm


LABELS_KEY_NAME = "tricks"
AudParams = namedtuple('AudParams', ['window_size', 'step_size', 'sample_rate'])
SpectParams = namedtuple('SpectParams', ['img_width', 'img_height'])


def load_audio_sample(file_name):
    """
    load_audio_sample loads an audio sample from any file (mp3, wav, flac, etc.)

    It returns the audio as a numpy array, normalized, together with its frame-rate
    """
    sample = librosa.load(file_name, sr=48000)
    sample_fs = sample[1]
    norm_audio_data = sample[0] / (2**15)

    return norm_audio_data, sample_fs


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


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def match_labels_to_seq(segments, df: pd.DataFrame, audio_params: AudParams, spect_params: SpectParams, out_dir: str, class_mapping: dict):
    """
    The heart of the labeling:
    check if the the image contains a vehicle, convert to mel spectogram and save the image with the according label
    """
    for idx, seg in enumerate(tqdm(segments)):
        n_mels = spect_params.img_height # number of bins in spectrogram. Height of image
        hop_length = floor(audio_params.window_size * audio_params.sample_rate / spect_params.img_width)  # number of samples per time-step in spectrogram 

        mels = librosa.feature.melspectrogram(y=seg, sr=audio_params.sample_rate, n_mels=n_mels,
                                            n_fft=hop_length*4, hop_length=hop_length)
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)

        img = scale_minmax(mels, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        # img = 255-img # invert. make black==more energy

        class_codes = [0]
        threshold = audio_params.step_size / 4
        start = audio_params.step_size * idx
        end = start + audio_params.window_size

        # adding a threshold to the vehicle search
        thresh_start = start + threshold
        thresh_end = end - threshold

        df_parts = df[((df["start"] >= thresh_start) & (df["start"] <= thresh_end)) | ((df["end"] >= thresh_start) & (df["end"] <= thresh_end))]
        if len(df_parts) > 0:
            # class_codes = (df_parts['labels'].cat.codes + 1).to_list()
            class_codes = list(map(lambda x: class_mapping[x], (df_parts['labels'].cat.codes + 1).to_list()))

        im = Image.fromarray(img)
        for class_code in class_codes:
            filename = join(out_dir, f"img_{idx}_{start}_{end}_{class_code}.jpeg")
            makedirs(dirname(filename), exist_ok=True)
            im.save(filename)


def main(args):
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    _get_labels = partial(get_labels, label_file=args.label_json)

    spect_params = SpectParams(img_width=params["img_width"], 
                               img_height=params['img_height'])

    for idx, audio_file in enumerate(glob(join(args.in_dir, "*.mp3"))):
        print(f" >>> Processing {audio_file} <<< ")
        audio_data, sample_fs = load_audio_sample(audio_file)
        audio_params = AudParams(window_size=params["window_size_in_sec"],
                    step_size=params["step_size_in_sec"],
                    sample_rate=sample_fs)

        segments = divide_sample(audio_data, audio_params)
        # processed_segments = remove_silent_segments(segments, params["silence_threshold"])

        df = _get_labels(file_name=audio_file)
        if df is None:
            import warnings
            warnings.warn(f"df was None for {audio_file}")
            continue

        match_labels_to_seq(segments, 
                            df, 
                            audio_params,
                            spect_params,
                            join(args.out_dir, str(idx)), 
                            params["class_mapping"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sound files and attach labels.')

    parser.add_argument('--in_dir', type=str, default="data/01_raw/", help='Folder with audio files to process')
    parser.add_argument('--label_json', type=str, default="data/02_labels/labels.json", help='Label file')
    parser.add_argument('--out_dir', type=str, default="data/03_primary/three_cat/", help='images output folder')

    args = parser.parse_args()

    main(args)
