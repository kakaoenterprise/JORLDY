import cv2 
import numpy as np

class ImgProcessor:
    def __init__(self):
        pass        
    
    def convert_img(self, img, gray_img, img_width, img_height):
        img = cv2.resize(img, dsize=(img_width, img_height))
        if gray_img:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2,0,1))
        return img