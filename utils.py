# -*- coding: utf-8 -*-

from glob import glob
import random

import scipy.misc
import skimage.transform
import skimage.io
import numpy as np

def data_generator(file_list, batch_size):
    idx = 0
    list_length = len(file_list)
    while True:
        batch = []
        for _ in range(batch_size):
            if idx >= list_length:
                idx = 0
                random.shuffle(file_list)
            batch.append(file_list[idx])
            idx += 1
        yield batch

def get_image(file_path, input_hw, is_random_clip=True, is_random_flip=True):
    """
    画像の読み込み・サイズ変更・データ拡張を行うメソッドです。
    :param file_path: str
        画像ファイルのパス。
    :param input_hw: int
        出力画像の画像の1辺の大きさ。
    :param is_random_clip: bool
        画像からランダムな領域切り出しを行うかどうか。
    :param is_random_flip: bool
        画像にランダムに水平方向の反転を加えるかどうか。
    :return: ndarray
        出力画像。
    """
    image = imread(file_path)
    h, w, c = image.shape
    if min(h, w) > input_hw *2:
        if h <= w:
            image = skimage.transform.resize(image, [input_hw * 2, input_hw * 2 * w // h])
        else:
            image = skimage.transform.resize(image, [input_hw * 2 * h // w, input_hw * 2])
    if is_random_clip and (h >= input_hw) and (w >= input_hw):
        image = random_clip(image, input_hw)
    else:
        image = center_crop(image)
        image = skimage.transform.resize(image, [input_hw, input_hw])

    is_flip = random.choice([True, False])
    if is_random_flip and is_flip:
        image = image[:, ::-1, :]

    return image


def imread(file_path):
    """
    画像の読み込みを行うメソッドです。
    :param file_path: str
        入力画像のファイルパス。
    :return: ndarray
        読み込んだ画像を [-1, 1]に規格化して返します。
    """
    image = skimage.io.imread(file_path).astype(np.float32)
    return image / 127.5 - 1 #0→255を-1→1に変換

def random_clip(image, input_hw):
    """
    画像からランダムな正方形領域を抽出するメソッドです。
    :param image: ndarray
        入力画像
    :param input_hw: int
        抽出する正方形の1辺の長さ
    :return: ndarray
        出力画像
    """
    h ,w, c = image.shape
    if h == input_hw:
        random_y = 0
    else:
        random_y = random.randint(0, h - input_hw)

    if w == input_hw:
        random_x = 0
    else:
        random_x = random.randint(0, w - input_hw)
    return image[random_y:random_y+input_hw, random_x:random_x+input_hw, :]


def center_crop(image):
    """
    画像から可能な限り大きな正方形領域を抽出するメソッドです。
    入力画像の中央から取得します。
    :param image: ndarray
        入力画像。
    :return: ndarray
        抽出後の出力画像。
    """
    h, w, c = image.shape
    if h >= w:
        crop_wh = w
        sub = int((h - w) // 2)
        trimmed = image[sub:sub+crop_wh, :, :]
    else:
        crop_wh = h
        sub = int((w - h) // 2)
        trimmed = image[:, sub:sub+crop_wh, :]
    return trimmed


def output_sample_image(path, combine_image):
    """
    画像を出力するメソッドです。
    :param path: str
        出力先ファイルパス。
    :param combine_image:  ndarray
        出力対象の画像です。[-1, 1] に規格化されているものを想定しています。
    :return: None
    """
    image = (combine_image+1) * 127.5
    skimage.io.imsave(path, image.astype(np.uint8))