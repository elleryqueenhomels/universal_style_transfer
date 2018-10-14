# Test the trained decoder
# Use the Encoder-Decoder to reconstruct image

import tensorflow as tf

from style_transfer_net import StyleTransferNet
from utils import list_images, get_images, save_single_image


ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
MODEL_SAVE_PATH = 'models/autoencoder'
TEST_IMG_DIR = 'test/input'
OUTPUT_DIR = 'test/output'


def test_autoencoder(autoencoder_levels, model_save_path):
    
    input_imgs_paths = list_images(TEST_IMG_DIR)

    with tf.Graph().as_default(), tf.Session() as sess:

        input_img = tf.placeholder(
            tf.float32, shape=(1, None, None, 3), name='input_img')

        stn = StyleTransferNet(ENCODER_WEIGHTS_PATH, autoencoder_levels)

        input_encs = [encoder.encode(input_img)[0] for encoder in stn.encoders]

        output_imgs = [decoder.decode(input_enc) for decoder, input_enc in zip(stn.decoders, input_encs)]

        sess.run(tf.global_variables_initializer())

        # restore the trained model and run the reconstruction
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, model_save_path)

        for input_img_path in input_imgs_paths:

            img = get_images(input_img_path)

            for autoencoder_id, output_img in zip(autoencoder_levels, output_imgs):

                out = sess.run(output_img, feed_dict={input_img: img})

                save_single_image(out[0], input_img_path, OUTPUT_DIR, prefix=autoencoder_id + '-')


def main():
    autoencoder_levels = [1, 2, 3, 4, 5]
    model_save_path = MODEL_SAVE_PATH + '-done'

    test_autoencoder(autoencoder_levels, model_save_path)

    print('\n>>>>> Testing all done!\n')


if __name__ == '__main__':
    main()

