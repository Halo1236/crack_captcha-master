#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2019/7/12 14:38
"""
import glob

import numpy as np
from keras.applications.xception import preprocess_input
# from skimage import transform
from scipy import misc
import time
from keras.models import load_model
import requests
import os
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

dir = os.path.dirname(__file__)
h5_file = os.path.join(dir, 'CaptchaForWechat.h5')
model = load_model(h5_file)

url = "http://mp.weixin.qq.com/mp/verifycode?cert=1563158913054.2224"
img_size = (50, 120)
letter_list = [chr(i) for i in range(97, 123)]


def captcha_predict(url):
    """
    :params:url
    :return:captcha(string)
    """
    x = []
    t1 = time.time()
    try:
        print(url)
        img = requests.get(url)
        with open('code.jpg', 'wb') as f:
            f.write(img.content)
        all_img = glob.glob(r'./*.jpg')
        image = mpimg.imread(all_img[0])
        plt.axis('off')
        plt.imshow(image)
        plt.show()

        x.append(misc.imresize(misc.imread(all_img[0]), img_size))
        x = preprocess_input(np.array(x).astype(float))
        z = model.predict(x)
        z = np.array([i.argmax(axis=1) for i in z]).T
        result = z.tolist()
        v = ''.join([letter_list[i] for i in result[0]])
    except Exception as e:
        v = "error"
        print(e)
    t2 = time.time()
    print("总耗时： {:.2f} s".format(t2 - t1))
    data = {
        "captcha": v,
        "runtime": t2 - t1
    }
    print(data)
    return data


if __name__ == '__main__':
    while True:
        captcha_predict(url)
