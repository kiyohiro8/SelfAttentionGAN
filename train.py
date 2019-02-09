# -*- coding: utf-8 -*-

from config import Config
from SAGAN import SAGAN

if __name__ == "__main__":
    config = Config()
    model = SAGAN(config)
    if config.RESUME_TRAIN:
        discriminator_path = "./result/181104_1905/weights/ramen_cam/discriminator245000.hdf5"
        generator_path = "./result/181104_1905/weights/ramen_cam/generator245000.hdf5"
        print("Training start at {} iterations".format(config.COUNTER))
        model.resume_train(discriminator_path, generator_path, config.COUNTER)
    else:
        print("Training start")
        model.train()