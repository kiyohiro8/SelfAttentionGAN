# -*- coding: utf-8 -*-

import time
import datetime
import os
import random
from glob import glob

import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input
from keras import optimizers
import keras.backend as K

import utils
import net_utils


class SAGAN():

    def __init__(self, config):
        self.config = config

    def build_model(self, discriminator_path=None, generator_path=None):

        if discriminator_path:
            self.discriminator = load_model(discriminator_path)
        else:
            self.discriminator = net_utils.discriminator_SN(self.config.IMAGE_SHAPE, base_name="discriminator",
                                                     use_res=self.config.USE_RES)
        if generator_path:
            self.generator = load_model(generator_path)
        else:
            self.generator = net_utils.generator_SN(self.config.LATENT_DIM, self.config.IMAGE_SHAPE,
                                             self.config.NUMBER_RESIDUAL_BLOCKS, base_name="generator")
            #self.generator = net_utils.generator(self.config.LATENT_DIM, self.config.IMAGE_SHAPE,
            #                                 self.config.NUMBER_RESIDUAL_BLOCKS, base_name="generator")

        D_real_input = Input(shape=self.config.IMAGE_SHAPE)
        noise_vector = Input(shape=(self.config.LATENT_DIM, ))
        D_fake_input = self.generator(noise_vector)

        epsilon = K.placeholder(shape=(None, 1, 1, 1))
        D_merged_input = Input(shape=self.config.IMAGE_SHAPE,
                                    tensor=epsilon * D_real_input
                                           + (1 - epsilon) * D_fake_input)
        loss_real = K.mean(self.discriminator(D_real_input))
        loss_fake = K.mean(self.discriminator(D_fake_input))

        grad_mixed = K.gradients(self.discriminator(D_merged_input), [D_merged_input])[0]
        norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
        grad_penalty = K.mean(K.square(norm_grad_mixed - 1))

        loss_d = loss_fake - loss_real + self.config.LAMBDA * grad_penalty

        self.optimizer_d = optimizers.Adam(lr=self.config.D_LEARNING_RATE,
                                    beta_1=self.config.BETA_1,
                                    beta_2=self.config.BETA_2)

        D_training_updates = self.optimizer_d.get_updates(self.discriminator.trainable_weights,[],loss_d)

        self.D_train = K.function([D_real_input, noise_vector, epsilon],
                                  [loss_real, loss_fake],
                                  D_training_updates)

        self.optimizer_g = optimizers.Adam(lr=self.config.G_LEARNING_RATE,
                                      beta_1=self.config.BETA_1,
                                      beta_2=self.config.BETA_2)

        loss_g = - loss_fake
        G_training_updates = self.optimizer_g.get_updates(self.generator.trainable_weights,[],loss_g)
        self.G_train = K.function([noise_vector], [loss_g], G_training_updates)

    def train(self):
        self.build_model()
        print(self.generator.layers)
        print(self.discriminator.layers)
        self.train_iterations()

    def resume_train(self, discriminator_path, generator_path, counter):
        self.build_model(discriminator_path, generator_path)
        print(self.generator.layers)
        print(self.discriminator.layers)
        self.train_iterations(counter)

    def train_iterations(self, counter=0):

        now = datetime.datetime.now()
        datetime_sequence = "{0}{1:02d}{2:02d}_{3:02}{4:02d}".format(str(now.year)[-2:], now.month, now.day ,
                                                                    now.hour, now.minute)
        file_list = glob(os.path.join(self.config.DATA_DIR, self.config.DATASET, self.config.DATA_EXT))

        random.seed(42)
        random.shuffle(file_list)

        val_ratio = 0.1
        train_file_list = file_list[round(len(file_list) * val_ratio):]
        val_file_list = file_list[:round(len(file_list) * val_ratio)]

        dataset = utils.data_generator(train_file_list, self.config.BATCH_SIZE)

        experiment_dir = os.path.join(self.config.RESULT_DIR, datetime_sequence)

        sample_output_dir = os.path.join(experiment_dir, "sample", self.config.DATASET)
        weights_output_dir = os.path.join(experiment_dir, "weights", self.config.DATASET)
        weights_output_dir_resume = os.path.join(experiment_dir, "weights", "resume")

        os.makedirs(sample_output_dir, exist_ok=True)
        os.makedirs(weights_output_dir, exist_ok=True)
        os.makedirs(weights_output_dir_resume, exist_ok=True)

        self.config.output_config(os.path.join(experiment_dir, "config.txt"))

        start_time = time.time()
        met_curve = pd.DataFrame(columns=["counter", "loss_d", "loss_d_real", "loss_d_fake",
                                          "loss_g"])

        train_val_curve = pd.DataFrame(columns=["counter", "train_loss_d", "val_loss_d"])

        h, w, c = self.config.IMAGE_SHAPE
        number_samples = (1024 // h) * (1024 // w)
        fixed_noise = np.random.normal(size=(number_samples, self.config.LATENT_DIM)).astype('float32')

        for epoch in range(self.config.EPOCH):
            for iter in range(self.config.ITER_PER_EPOCH):
                for _ in range(self.config.NUM_CRITICS):
                    batch_files = next(dataset)
                    real_batch = np.array([utils.get_image(file, input_hw=self.config.IMAGE_SHAPE[0])
                                           for file in batch_files])

                    noise = np.random.normal(size=(self.config.BATCH_SIZE, self.config.LATENT_DIM))
                    epsilon = np.random.uniform(size=(self.config.BATCH_SIZE, 1, 1, 1))
                    errD_real, errD_fake = self.D_train([real_batch, noise, epsilon])
                    errD = errD_real - errD_fake

                noise = np.random.normal(size=(self.config.BATCH_SIZE, self.config.LATENT_DIM))
                errG, = self.G_train([noise])

                elapsed = time.time() - start_time

                print("epoch {0} {1}/{2} loss_d:{3:.4f} loss_d_real:{4:.4f} "
                      "loss_d_fake:{5:.4f}, loss_g:{6:.4f}, {7:.2f}秒".
                      format(epoch, iter, 1000, errD, errD_real, errD_fake, errG, elapsed))

                if counter % 10 == 0:
                    temp_df = pd.DataFrame({"counter":[counter], "loss_d":[errD],
                                            "loss_d_real":[errD_real], "loss_d_fake":[errD_fake],
                                            "loss_g":[errG]})
                    met_curve = pd.concat([met_curve, temp_df], axis=0)

                if counter % 500 == 0:

                    # validation lossの計算
                    val_D_real = 0
                    val_D_fake = 0

                    val_size = len(val_file_list)
                    for i in range(val_size//self.config.BATCH_SIZE):
                        val_files = val_file_list[i*self.config.BATCH_SIZE:(i+1)*self.config.BATCH_SIZE]
                        val_batch = np.array([utils.get_image(file, input_hw=self.config.IMAGE_SHAPE[0])
                                           for file in val_files])
                        val_D_real += np.mean(self.discriminator.predict(val_batch))
                        noise = np.random.normal(size=(self.config.BATCH_SIZE, self.config.LATENT_DIM))
                        val_D_fake += np.mean(self.discriminator.predict(self.generator.predict(noise)))
                    if not val_size % self.config.BATCH_SIZE == 0:
                        val_files = val_file_list[-val_size%self.config.BATCH_SIZE:]
                        val_batch = np.array([utils.get_image(file, input_hw=self.config.IMAGE_SHAPE[0])
                                           for file in val_files])
                        val_D_real += np.mean(self.discriminator.predict(val_batch))
                        noise = np.random.normal(size=(val_size%self.config.BATCH_SIZE, self.config.LATENT_DIM))
                        val_D_fake += np.mean(self.discriminator.predict(self.generator.predict(noise)))

                    val_loss = (val_D_real - val_D_fake) / val_size
                    temp_df = pd.DataFrame({"counter":[counter], "train_loss_d":[errD], "val_loss_d":[val_loss]})
                    train_val_curve = pd.concat([train_val_curve, temp_df], axis=0)
                    train_val_curve.to_csv(os.path.join(experiment_dir, self.config.DATASET+"_val.csv"), index=False)

                    # sample の出力
                    sample = self.generator.predict(fixed_noise)
                    sample_array = np.zeros(((1024 // h) * h, (1024 // w) * w, 3))
                    for n in range(number_samples):
                        i = n // (1024 // h)
                        j = n % (1024 // w)
                        sample_array[i*h:(i+1)*h, j*w:(j+1)*w, :] = sample[n, :, :, :]
                    file = "{0}_{1}.jpg".format(epoch, counter)
                    utils.output_sample_image(os.path.join(sample_output_dir, file), sample_array)

                if counter % 10000 == 0:
                    self.generator.save(os.path.join(weights_output_dir, "generator" + str(counter) + ".hdf5"))
                    self.discriminator.save(os.path.join(weights_output_dir, "discriminator" + str(counter) + ".hdf5"))
                    met_curve.to_csv(os.path.join(experiment_dir,
                                                  self.config.DATASET+".csv"), index=False)

                counter += 1

        sample = self.generator.predict(fixed_noise)
        h, w, c = self.config.IMAGE_SHAPE
        sample_array = np.zeros(((1024 // h) * h, (1024 // w) * w, 3))
        for n in range(number_samples):
            i = n // (1024 // h)
            j = n % (1024 // w)
            sample_array[i * h:(i + 1) * h, j * w:(j + 1) * w, 3] = sample[n, :, :, :]
        file = "{0}_{1}.jpg".format(self.config.EPOCH, counter)
        utils.output_sample_image(os.path.join(sample_output_dir, file), sample_array)

        self.generator.save(os.path.join(weights_output_dir, "generator" + str(counter) + ".hdf5"))
        self.discriminator.save(os.path.join(weights_output_dir, "discriminator" + str(counter) + ".hdf5"))
        met_curve.to_csv(os.path.join(experiment_dir,
                                      self.config.DATASET_A + "_"
                                      + self.config.DATASET_B + ".csv"),
                         index=False)


    def generate(self,weights_path, number_images=100):
        generator = net_utils.generator(self.config.LATENT_DIM, self.config.IMAGE_SHAPE,
                                             self.config.NUMBER_RESIDUAL_BLOCKS, base_name="generator")

        generator.load_weights(weights_path)

        now = datetime.datetime.now()
        datetime_sequence = "{0}{1:02d}{2:02d}_{3:02}{4:02d}".format(str(now.year)[-2:], now.month, now.day ,
                                                                    now.hour, now.minute)

        output_dir = os.path.join("generated", datetime_sequence)
        os.makedirs(output_dir, exist_ok=True)

        counter = 0

        while counter < number_images:
            noise = np.random.normal(size=(16, self.config.LATENT_DIM)).astype('float32')
            generated_images = generator.predict(noise)
            for i in range(16):
                file = "{}.jpg".format(counter)
                utils.output_sample_image(os.path.join(output_dir, file), generated_images[i, :, :, :])
                counter += 1
                if counter >= number_images:
                    break


