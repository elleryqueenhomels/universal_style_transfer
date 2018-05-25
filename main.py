# Demo - train the decoders & use them to stylize image

from __future__ import print_function

from train import train
from infer import stylize
from utils import list_images


IS_TRAINING = True

# for training
TRAINING_IMGS_PATH = 'MS_COCO'
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
MODEL_SAVE_PATH = 'models/autoencoder'
MODEL_SAVE_SUFFIX = '-done'
DEBUG = True
LOGGING_PERIOD = 10

# for inferring (stylize)
CONTENTS_DIR = 'images/content'
STYLES_DIR = 'images/style'
OUTPUT_DIR = 'output'
STYLE_RATIO = 0.6


def main():

    if IS_TRAINING:

        training_imgs_paths = list_images(TRAINING_IMGS_PATH)

        train(training_imgs_paths,
              ENCODER_WEIGHTS_PATH,
              MODEL_SAVE_PATH,
              debug=DEBUG,
              logging_period=LOGGING_PERIOD)

        
        print('\n>>>>>> Successfully done training...\n')

    else:

        contents_path = list_images(CONTENTS_DIR)
        styles_path = list_images(STYLES_DIR)
        model_path = MODEL_SAVE_PATH + MODEL_SAVE_SUFFIX

        stylize(contents_path, 
                styles_path, 
                OUTPUT_DIR, 
                ENCODER_WEIGHTS_PATH, 
                model_path, 
                style_ratio=STYLE_RATIO)

        print('\n>>>>>> Successfully done stylizing...\n')


if __name__ == '__main__':
    main()

