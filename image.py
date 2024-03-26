import numpy as np
import cv2
from functions import generate_color
from settings import Img_settings
from writing import Writing_style

class Image():
    def __init__(self, size: list, shape: int, colors: list):
        self.size = size
        self.shape = shape
        self.colors = colors
    
    def generate_empty(self):
        image = np.zeros((self.size[0], self.size[1], self.shape), np.uint8)
        color = generate_color(Img_settings.background_colors)
        image[:] = (color[0], color[1], color[2])
        self.img = image
    
    def write_text(self, style: Writing_style, text):
        cv2.putText(img=self.img,
                    text=text, 
                    org=style.coords, 
                    fontFace=style.font, 
                    fontScale=style.size,
                    color=style.color, 
                    thickness=style.thickness, 
                    lineType=cv2.LINE_AA)

    def save(self, path: str):
        cv2.imwrite(path, self.img)