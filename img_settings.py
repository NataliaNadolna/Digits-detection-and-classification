import cv2
import dataclasses

@dataclasses.dataclass()
class Img_settings:
    img_size = [28, 28]
    background_colors = dict(red = (180, 255), green = (180, 255), blue = (180, 255))
    ink_colors = dict(red = (0,100), green = (0,100), blue = (0,100))
    number_position = dict(left = 3, right = 6, down = 24, up = 24)
    font_scale = [0.9, 1.0]
    thickness = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    fonts = [cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX, 
            cv2.FONT_HERSHEY_COMPLEX, 
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_ITALIC]
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']