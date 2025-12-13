import cv2
import numpy as np
import requests


def load_image(image_url):
    """
    支持 URL 和本地路径
    """
    if image_url.startswith("http"):
        resp = requests.get(image_url, timeout=10)
        img_array = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_url)

    if image is None:
        raise ValueError("无法读取图片：" + image_url)

    return image