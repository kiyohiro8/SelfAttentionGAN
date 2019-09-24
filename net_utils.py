# -*- coding: utf-8 -*-

import os

import numpy as np

from keras.layers import Input, Dense, Activation, Flatten, Add, Lambda, Concatenate, Reshape, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.engine.network import Network, Layer
from keras.initializers import TruncatedNormal
from keras.models import Model
import keras.backend as K

from SelfAttentionLayer import SelfAttention

from SpectralNormLayer import ConvSN2D, DenseSN


def set_trainable(model, prefix_list, trainable=False):
    for prefix in prefix_list:
        for layer in model.layers:
            if layer.name.startswith(prefix):
                layer.trainable = trainable
    return model


def generator(latent_dim, image_shape, num_res_blocks, base_name):
    initializer = TruncatedNormal(mean=0, stddev=0.2, seed=42)
    in_x = Input(shape=(latent_dim,))

    h, w, c = image_shape

    x = Dense(64*8*h//8*w//8, activation="relu", name=base_name+"_dense")(in_x)
    x = Reshape((h//8, w//8, -1))(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*4, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer,
                        name=base_name + "_conv1")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(x, training=1)
    x = Activation("relu")(x)


    # size//8→size//4→size//2→size
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*2, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer,
                        name=base_name + "_conv2")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(x, training=1)
    x = Activation("relu")(x)

    x = SelfAttention(ch=64*2, name=base_name+"_sa")(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*1, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_conv3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(x,training=1)
    x = Activation("relu")(x)
    x = UpSampling2D((2, 2))(x)
    out = Conv2D(3, kernel_size=3, strides=1, padding='same', activation="tanh",
                          kernel_initializer=initializer, name=base_name + "_out")(x)
    model = Model(in_x, out, name=base_name)
    return model


def generator_SN(latent_dim, image_shape, num_res_blocks, base_name):
    initializer = TruncatedNormal(mean=0, stddev=0.2, seed=42)
    in_x = Input(shape=(latent_dim,))

    h, w, c = image_shape

    x = Dense(64*8*h//8*w//8, activation="relu", name=base_name+"_dense")(in_x)
    x = Reshape((h//8, w//8, -1))(x)

    x = UpSampling2D((2, 2))(x)
    x = ConvSN2D(64*4, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer,
                        name=base_name + "_conv1")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(x, training=1)
    x = Activation("relu")(x)

    # size//8→size//4→size//2→size
    x = UpSampling2D((2, 2))(x)
    x = ConvSN2D(64*2, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer,
                        name=base_name + "_conv2")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(x, training=1)
    x = Activation("relu")(x)

    x = SelfAttention(ch=64*2, name=base_name+"_sa")(x)

    x = UpSampling2D((2, 2))(x)
    x = ConvSN2D(64*1, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_conv3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(x,training=1)
    x = Activation("relu")(x)
    x = UpSampling2D((2, 2))(x)
    out = ConvSN2D(3, kernel_size=3, strides=1, padding='same', activation="tanh",
                          kernel_initializer=initializer, name=base_name + "_out")(x)
    model = Model(in_x, out, name=base_name)
    return model


def discriminator(input_shape, base_name, num_res_blocks=0,is_D=True, use_res=False):
    initializer_d = TruncatedNormal(mean=0, stddev=0.1, seed=42)

    D = in_D = Input(shape=input_shape)
    D = Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv1")(D)

    D = LeakyReLU(0.2)(D)

    D = SelfAttention(ch=64, name=base_name+"_sa")(D)

    D = Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv2")(D)

    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(D, training=1)
    D = LeakyReLU(0.2)(D)

    D = Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv3")(D)
    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(D, training=1)
    D = LeakyReLU(0.2)(D)

    D = Conv2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv4")(D)

    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(D, training=1)
    D = LeakyReLU(0.2)(D)
    D = Conv2D(1, kernel_size=1, strides=1, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv5")(D)

    D = Flatten()(D)
    out = Dense(units=1, activation=None, name=base_name + "_out")(D)
    model = Model(in_D, out, name=base_name)

    return model


def discriminator_SN(input_shape, base_name, num_res_blocks=0,is_D=True, use_res=False):
    initializer_d = TruncatedNormal(mean=0, stddev=0.1, seed=42)

    D = in_D = Input(shape=input_shape)
    D = ConvSN2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv1")(D)

    D = LeakyReLU(0.2)(D)

    D = SelfAttention(ch=64, name=base_name+"_sa")(D)

    D = ConvSN2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv2")(D)

    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(D, training=1)
    D = LeakyReLU(0.2)(D)

    D = ConvSN2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv3")(D)
    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(D, training=1)
    D = LeakyReLU(0.2)(D)


    D = ConvSN2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv4")(D)

    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(D, training=1)
    D = LeakyReLU(0.2)(D)
    D = ConvSN2D(1, kernel_size=1, strides=1, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv5")(D)

    D = Flatten()(D)
    out = DenseSN(units=1, activation=None, name=base_name + "_out")(D)
    model = Model(in_D, out, name=base_name)

    return model


def residual_block(x, base_name, block_num, initializer, num_channels=128,is_D=False):
    y = Conv2D(num_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv1")(x)
    if not is_D:
        y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn1")(y, training=1)
    y = Activation("relu")(y)
    y = Conv2D(num_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv2")(y)
    if not is_D:
        y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn2")(y, training=1)
    return Add()([x, y])


def save_weights(model, path, counter, base_name=""):
    filename = base_name +str(counter) + ".hdf5"
    output_path = os.path.join(path, filename)
    model.save_weights(output_path)