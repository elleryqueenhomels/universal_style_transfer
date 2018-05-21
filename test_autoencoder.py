# Test the trained decoder
# Use the Encoder-Decoder to reconstruct image

import tensorflow as tf

from style_transfer_net import StyleTransferNet
from utils import list_images, get_images, save_single_image


ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
MODEL_SAVE_PATH = 'models/autoencoder'
TEST_IMG_DIR = 'test/input'
OUTPUT_DIR = 'test/output'


def main(autoencoder_id, model_save_suffix):
    
    input_imgs_paths = list_images(TEST_IMG_DIR)
    model_save_path  = '%s_%d-%s' % (MODEL_SAVE_PATH, autoencoder_id, model_save_suffix)

    with tf.Graph().as_default(), tf.Session() as sess:
        input_img = tf.placeholder(
            tf.float32, shape=(1, None, None, 3), name='input_img')

        stn = StyleTransferNet(ENCODER_WEIGHTS_PATH)

        input_enc = stn.encoders[autoencoder_id - 1].encode(input_img)

        output_img = stn.decoders[autoencoder_id - 1].decode(input_enc)

        sess.run(tf.global_variables_initializer())

        # restore the trained model and run the reconstruction
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, model_save_path)

        for input_img_path in input_imgs_paths:
            img = get_images(input_img_path)

            out = sess.run(output_img, feed_dict={input_img: img})

            save_single_image(out[0], input_img_path, OUTPUT_DIR)


if __name__ == '__main__':

    main(5, 20000)

    print("\n>>>>> Test finished!\n")

