import cv2
from roboflow import Roboflow
from frame import Frame
from models import *
import torch

def main():

    cap = cv2.VideoCapture('animation.gif')

    rf = Roboflow(api_key="zYTszjLxeYF4ctnj9MbI")
    project = rf.workspace().project("dog-8syau")
    detection_model = project.version(2).model

    classification_model_path = 'C:/Users/Natalia/Desktop/PWr/2/NTDD/programowanie/_Digits Detection and Clasiffication/model.pth'
    classification_model = MNISTModel(input_shape=3, hidden_units=10, output_shape=10) 
    classification_model.load_state_dict(torch.load(f=classification_model_path))

    if (cap.isOpened() == False): 
        print("Error opening video stream or file")

    iteration = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        iteration += 1
        if ret == True:
            image = Frame(frame, "frame.jpg")
            detection = image.detect_digits(detection_model)
            first_number = image.find_first_number(detection)
            coords_list = image.find_coords(first_number)
            number = image.crop_digits(coords_list)
            num = image.classify_number(classification_model, number)
            image.calculate_speed(number = num, frame = iteration-1)

            cv2.imshow('Frame', frame)
    
            if cv2.waitKey(700) & 0xFF == ord('q'):
                break
        else: 
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

