#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2019/7/16 17:35
"""
from flask import Flask
import glob
import asyncio
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

app = Flask(__name__)

dir = os.path.dirname(__file__)
h5_file = os.path.join(dir, 'CaptchaForWechat.h5')
model = load_model(h5_file)

url = "http://mp.weixin.qq.com/mp/verifycode?cert=%s" % time.time()
img_size = (50, 120)
letter_list = [chr(i) for i in range(97, 123)]


async def captcha_predict():
    """
    :params:url
    :return:captcha(string)
    """
    x = []
    v = ''
    ret = ''
    t1 = time.time()
    url = "http://mp.weixin.qq.com/mp/verifycode?cert=%s" % time.time()
    try:
        print(url)
        session = requests.session()
        img = session.get(url)

        with open('code.jpg', 'wb') as f:
            f.write(img.content)
        all_img = glob.glob(r'./*.jpg')
        # image = mpimg.imread(all_img[0])
        # plt.axis('off')
        # plt.imshow(image)
        # plt.show()

        x.append(misc.imresize(misc.imread(all_img[0]), img_size))
        x = preprocess_input(np.array(x).astype(float))
        z = model.predict(x)
        z = np.array([i.argmax(axis=1) for i in z]).T
        result = z.tolist()
        v = ''.join([letter_list[i] for i in result[0]])
        ret = await post(session, v, url)
    except Exception as e:

        print(e)
    t2 = time.time()
    print("总耗时： {:.2f} s".format(t2 - t1))
    data = {
        "captcha": v,
        "runtime": t2 - t1,
        'url': url
    }
    print(data)
    return str(ret)


def post(session, code, url):
    unlock_url = 'https://mp.weixin.qq.com/mp/verifycode'
    data = {
        'cert': time.time() * 1000,
        'input': code
    }
    headers = {
        'Host': 'mp.weixin.qq.com',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Referer': url
    }
    r_unlock = session.post(unlock_url, data, headers=headers)
    print(r_unlock.json())
    return r_unlock.json()


def runEventLoop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(captcha_predict())
    loop.close()


@app.route('/get')
def get_code():
    oldloop = asyncio.get_event_loop()
    runEventLoop()
    asyncio.set_event_loop(oldloop)
    return 'ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
