# -*- coding: utf-8 -*-

from config import Config
from WGANgp import  WGANgp

if __name__ == "__main__":
    config = Config()
    model = WGANgp(config)
    model.generate(config.MODEL_FILE_PATH, 100)