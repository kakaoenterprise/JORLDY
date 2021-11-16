import cv2
import numpy as np


class ImgProcessor:
    def __init__(self, gray_img, img_width, img_height):
        self.gray_img = gray_img
        self.img_width = img_width
        self.img_height = img_height

    def convert_img(self, img):
        img = cv2.resize(img, dsize=(self.img_width, self.img_height))
        if self.gray_img:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)
        return img
