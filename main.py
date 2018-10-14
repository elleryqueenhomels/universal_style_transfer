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
AUTUENCODER_LEVELS_TRAIN = [5, 4, 3, 2, 1]

# for inferring (stylize)
CONTENTS_DIR = 'images/content'
STYLES_DIR = 'images/style'
OUTPUT_DIR = 'outputs'

STYLE_RATIO = 0.8
REPEAT_PIPELINE = 1
AUTUENCODER_LEVELS_INFER = [3, 2, 1]


def main():

    if IS_TRAINING:
        training_imgs_paths = list_images(TRAINING_IMGS_PATH)

        train(training_imgs_paths,
              ENCODER_WEIGHTS_PATH,
              MODEL_SAVE_PATH,
              autoencoder_levels=AUTUENCODER_LEVELS_TRAIN,
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
                style_ratio=STYLE_RATIO,
                repeat_pipeline=REPEAT_PIPELINE,
                autoencoder_levels=AUTUENCODER_LEVELS_INFER)

        print('\n>>>>>> Successfully done stylizing...\n')


if __name__ == '__main__':
    main()

