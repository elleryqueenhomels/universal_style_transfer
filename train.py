# Train the Auto-Encoder

from __future__ import print_function

import numpy as np
import tensorflow as tf

from decoder import Decoder
from encoder import Encoder
from style_transfer_net import StyleTransferNet
from utils import get_images, get_training_images


# (height, width, channels)
TRAINING_IMAGE_SHAPE = (256, 256, 3)

PIXEL_LOSS_WEIGHT = 1
FEATURE_LOSS_WEIGHT = 1

EPOCHS = 2
EPSILON = 1e-5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4


def train(training_imgs_paths, encoder_weights_path, model_save_path, debug=False, logging_period=100):
    if debug:
        from datetime import datetime
        start_time = datetime.now()

    # guarantee the number of training imgs is a multiple of BATCH_SIZE
    mod = len(training_imgs_paths) % BATCH_SIZE
    if mod > 0:
        print('Train set has been trimmed %d samples...' % mod)
        training_imgs_paths = training_imgs_paths[:-mod]

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:

        # create encoders & decoders through StyleTransferNet
        stn = StyleTransferNet(encoder_weights_path)

        # initialize all the variables
        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        for index, (encoder, decoder) in enumerate(zip(stn.encoders, stn.decoders)):

            autoencoder_id = 5 - index

            input_imgs = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='input_imgs_%d' % autoencoder_id)

            # logic: input_img -> encode() -> img_features -> decode() -> output_img
            input_features = encoder.encode(input_imgs)
            output_imgs = decoder.decode(input_features)
            output_features = encoder.encode(output_imgs)

            # compute the pixel loss
            pixel_loss = tf.losses.mean_squared_error(input_imgs, output_imgs)

            # compute the feature loss
            feature_loss = tf.losses.mean_squared_error(input_features, output_features)

            # total loss
            total_loss = PIXEL_LOSS_WEIGHT * pixel_loss + FEATURE_LOSS_WEIGHT * feature_loss

            # Training step
            trainer = tf.train.AdamOptimizer(LEARNING_RATE)
            train_op = trainer.minimize(total_loss)
            trainer_initializers = [var.initializer for var in trainer.variables()]

            sess.run(trainer_initializers)

            """ Start Training """
            step = 0
            n_batches = int(len(training_imgs_paths) // BATCH_SIZE)

            if debug:
                elapsed_time = datetime.now() - start_time
                print('\nElapsed time for preprocessing before actually train the Decoder_%d: %s' % (autoencoder_id, elapsed_time))
                print('Now begin to train the Decoder_%d...\n' % autoencoder_id)
                start_time = datetime.now()

            try:
                for epoch in range(EPOCHS):

                    np.random.shuffle(training_imgs_paths)

                    for batch in range(n_batches):
                        # retrive a batch of trainging images
                        img_batch_paths = training_imgs_paths[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

                        # img_batch = get_images(img_batch_paths, height=HEIGHT, width=WIDTH)
                        img_batch = get_training_images(img_batch_paths, crop_height=HEIGHT, crop_width=WIDTH)

                        # run the training step
                        sess.run(train_op, feed_dict={input_imgs: img_batch})

                        step += 1

                        if step % 1000 == 0:
                            saver.save(sess, '%s_%d' % (model_save_path, autoencoder_id), 
                                global_step=step, write_meta_graph=False)

                        if debug:
                            is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                            if is_last_step or step == 1 or step % logging_period == 0:
                                elapsed_time = datetime.now() - start_time
                                _pixel_loss, _feature_loss, _loss = sess.run([pixel_loss, feature_loss, total_loss], 
                                    feed_dict={input_imgs: img_batch})

                                print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
                                print('pixel   loss: %.3f' % (_pixel_loss))
                                print('feature loss: %.3f' % (_feature_loss))
                                print('total   loss: %.3f' % (_loss))
                                print('\n')

                # finish training current decoder, save the model
                saver.save(sess, '%s_%d' % (model_save_path, autoencoder_id), global_step=step)

                if debug:
                    elapsed_time = datetime.now() - start_time
                    print('>>> Successfully training decoder_%d! Elapsed time: %s\n' % (autoencoder_id, elapsed_time))

            except:
                saver.save(sess, '%s_%d' % (model_save_path, autoencoder_id), global_step=step)
                print('\nSomething wrong happens! Current model is saved with current step: %d\n' % step)

                if debug:
                    elapsed_time = datetime.now() - start_time
                    print('Elapsed time: %s\n' % elapsed_time)

                exit()

        """ Done all trainings & Save the final model """
        saver.save(sess, model_save_path + '-done')

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % model_save_path + '-done')

