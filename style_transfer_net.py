# Style Transfer Network
# Architecture:
# input -> encoder5 -> WCT -> decoder5 ->
#          encoder4 -> WCT -> decoder4 ->
#          encoder3 -> WCT -> decoder3 ->
#          encoder2 -> WCT -> decoder2 ->
#          encoder1 -> WCT -> decoder1 -> result

import tensorflow as tf

from decoder import Decoder
from encoder import Encoder


ENCODER_5 = 5
DECODER_5 = (
    'conv5_1', 
    'conv4_4', 'conv4_3', 'conv4_2', 'conv4_1',
    'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1',
    'conv2_2', 'conv2_1',
    'conv1_2', 'conv1_1',
)

ENCODER_4 = 4
DECODER_4 = (
    'conv4_1',
    'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1',
    'conv2_2', 'conv2_1',
    'conv1_2', 'conv1_1',
)

ENCODER_3 = 3
DECODER_3 = (
    'conv3_1',
    'conv2_2', 'conv2_1',
    'conv1_2', 'conv1_1',
)

ENCODER_2 = 2
DECODER_2 = (
    'conv2_1',
    'conv1_2', 'conv1_1',
)

ENCODER_1 = 1
DECODER_1 = (
    'conv1_1',
)

AUTOENCODERS = (
    (ENCODER_5, DECODER_5),
    (ENCODER_4, DECODER_4),
    (ENCODER_3, DECODER_3),
    (ENCODER_2, DECODER_2),
    (ENCODER_1, DECODER_1),
)


class StyleTransferNet(object):

    def __init__(self, encoder_weights_path):
        self.encoders = []
        self.decoders = []

        for autoencoder in AUTOENCODERS:
            self.encoders.append(Encoder(encoder_weights_path, autoencoder[0]))
            self.decoders.append(Decoder(autoencoder[1]))

    def transform(self, content, style, style_ratio):
        # assume the shape of content and style both are 1xHxWxC

        output = content

        for enc, dec in zip(self.encoders, self.decoders):
            content_enc, _ = enc.encode(output)
            style_enc, _   = enc.encode(style)

            synthesis = self._wct(content_enc, style_enc, style_ratio)

            output = dec.decode(synthesis)

        return output

    def _wct(self, content, style, style_ratio, eps=1e-8):
        # Remove batch dim and reorder to CxHxW
        content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
        style_t   = tf.transpose(tf.squeeze(style)  , (2, 0, 1))

        # Unpack to get each dim (Channel, Height, Width)
        Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
        Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

        # CxHxW -> Cx(H*W)
        content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
        style_flat   = tf.reshape(style_t,   (Cs, Hs*Ws))

        # Content covariance
        mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
        fc = content_flat - mc
        fc_cov = tf.matmul(fc, fc, transpose_b=True) / tf.cast(Hc*Wc - 1, tf.float32) + tf.eye(Cc) * eps

        # Style covariance
        ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
        fs = style_flat - ms
        fs_cov = tf.matmul(fs, fs, transpose_b=True) / tf.cast(Hs*Ws - 1, tf.float32) + tf.eye(Cs) * eps

        # tf.svd is slower on GPU
        with tf.device('/cpu:0'):
            Sc, Uc, _ = tf.svd(fc_cov)
            Ss, Us, _ = tf.svd(fs_cov)

        # Filter small singular values
        k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
        k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

        # Whitening content feature
        Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
        fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:, :k_c], transpose_b=True), fc)

        # Coloring content with style
        Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
        fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:, :k_s], transpose_b=True), fc_hat)

        # Re-center with mean of style
        fcs_hat = fcs_hat + ms

        # Blend whiten-colored feature with original content feature
        blended = style_ratio * fcs_hat + (1 - style_ratio) * (fc + mc)

        # Cx(H*W) -> CxHxW
        blended = tf.reshape(blended, (Cc, Hc, Wc))
        # CxHxW -> 1xHxWxC
        blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)

        return blended

