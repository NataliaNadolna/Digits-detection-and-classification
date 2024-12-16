from digits import Digit, Number
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import torchvision, torch
from torchvision import transforms

class Frame():
    def __init__(self, img: np.ndarray, img_path: str):
        self.img = img
        self.img_path = img_path
        cv2.imwrite(img_path, img)

    def detect_digits(self, model):
        detection = model.predict(self.img_path, confidence=40, overlap=30).json()
        model.predict(self.img_path, confidence=40, overlap=30).save("./main/program_files/prediction.jpg")
        prediction_img = cv2.imread('./main/program_files/prediction.jpg')
        return detection

    def find_first_number(self, detection):
        number = []
        for index in range(len(detection['predictions'])):
            box = detection['predictions'][index]
            if box['x'] < 115:
                number.append(box)
        return number
    
    def find_coords(self, number):
        coords_list = []
        for bounding_box in number:
            x1 = bounding_box['x'] - bounding_box['width'] / 2
            x2 = bounding_box['x'] + bounding_box['width'] / 2
            y1 = bounding_box['y'] - bounding_box['height'] / 2
            y2 = bounding_box['y'] + bounding_box['height'] / 2
            coords = (x1, x2, y1, y2)
            coords_list.append(coords)
        coords_list.sort()
        return coords_list
    
    def crop_digits(self, coords_list):
        number = Number()

        for i, coords in enumerate(coords_list):
            x = 1
            digit_img = self.img[int(coords[2]-x):int(coords[3]+x), int(coords[0]-x):int(coords[1]+x)]

            path = ('./main/program_files/digit_%02i.jpg' %i)
            new_img = cv2.imwrite(path, digit_img)

            new_digit = Digit(new_img, path)
            number.append(new_digit)

        return number

    def calssify_digit(self, model, digit: Digit):
        image = torchvision.io.read_image(str(digit.path)).type(torch.float32)
        image = image / 255.

        img_transform = transforms.Compose([
            transforms.Resize((28, 28))])
        
        img_transformed = img_transform(image)

        model.eval()
        img_pred = model(img_transformed.unsqueeze(dim=0))

        img_pred_probs = torch.softmax(img_pred, dim=1)
        pred_digit = torch.argmax(img_pred_probs, dim=1)
        return pred_digit.item()
    
    def convert(self,list):
        number = int("".join(map(str, list)))
        return number

    def classify_number(self, model, number: Number):
        digits_list = []
        for digit in number:
            pred = self.calssify_digit(model, digit)
            digits_list.append(pred)
        
        # print(f"Liczba: {self.convert(digits_list)}")
        return self.convert(digits_list)


    def calculate_speed(self, number, frame):
        if frame == 0:
            frame = 1
        print(f"Distance: {number} \t Speed: {number} / {frame} = {number/frame}")





  