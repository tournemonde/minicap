import tensorflow as tf
import numpy as np
import os
import argparse
from glob import glob
from yaml import safe_load


def process_image(image_path, image_wh):
    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
    image = tf.reshape(image, [image.shape[0], -1])
    image = channelize(image)
    image = tf.image.resize(image, [*image_wh])
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    return image


def channelize(x):
    return tf.stack([x, x, x], axis=-1)


def main(args):
    debug_prefix = ''
    params = safe_load(open(debug_prefix + "params.yaml"))["prepare"]
    img_wh = [params['img_width'], params['img_height']]
    # Load the TFLite model in TFLite Interpreter
    # ==== TFLITE_FILE_PATH = debug_prefix + 'data/06_models_tuned/mnet_tf23.tflite'
    interpreter = tf.lite.Interpreter(os.path.join(args.model_path, args.model))

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # input_shape = input_details[0]['shape']

    # === single image test ===
    # image_path = 'data/03_primary/three_cat/0/img_0_0.0_1.8_1.jpeg'
    # img = process_image(image_path, img_wh)
    # image_tensor = tf.expand_dims(img, axis=0)

    # === batch processing ===
    img_list = glob(os.path.join(args.image_path, '*.jpeg'))
    processed_image_list = [process_image(i, img_wh) for i in img_list]
    image_tensors = tf.stack(processed_image_list)

    interpreter.resize_tensor_input(input_details[0]['index'], image_tensors.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], image_tensors)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_preds = np.argmax(output_data, axis=-1)
    unique, counts = np.unique(y_preds, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(count_dict)


if __name__ == '__main__':
    debug_prefix = ''
    parser = argparse.ArgumentParser(description='Process sound files and attach labels.')
    parser.add_argument('--model_path', type=str, default=debug_prefix + 'data/06_models_tuned/',
                        help='path of the model')
    parser.add_argument('--model', type=str, help='name of the model to be tested')
    parser.add_argument('--image_path', type=str, default=debug_prefix + 'data/03_primary/inputs',
                        help='input directory with spectrograms')

    args = parser.parse_args()

    main(args)


