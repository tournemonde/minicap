import tensorflow as tf
import argparse
import os
from os.path import join
from tensorflow.keras.models import load_model
from yaml import safe_load

debug_prefix = ''
params = safe_load(open(debug_prefix + "params.yaml"))["train"]
NUM_CLASSES = params['num_classes']


def main(args):
    # print(os.listdir(args.model_path))
    # model = build_mnetv2(NUM_CLASSES)
    # model = model.load_weights(join(args.model_path, args.model))
    # converter = tf.lite.TFLiteConverter.from_saved_model(join(args.model_path, args.model))
    print(f'model name: {join(args.model_path, args.model)}')
    if args.model in os.listdir(args.model_path):
        print('exists')
    model = load_model(join(args.model_path, args.model))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(join(args.model_path, args.tflite_name), 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process sound files and attach labels.')
    parser.add_argument('--model_path', type=str, default=debug_prefix + 'data/06_models_tuned/', help='path of the model')
    parser.add_argument('--model', type=str, help='name of the model to be tested')
    parser.add_argument('--tflite_name', type=str, help='name of the output lite file')

    args = parser.parse_args()
    #args.model = '3class_VC_TF23_mnetv2_step2_3_1e-05_64_99.h5'
    #args.tflite_name = 'mnet_tf23.tflite'

    main(args)
