import cv2
from roboflow import Roboflow
from frame import Frame
from models import *
import torch
import os
from dotenv import load_dotenv

def main(video_path, detection_project_name, detection_project_version, classification_model_path, classification_model):

    cap = cv2.VideoCapture(video_path)

    # load detection model
    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(detection_project_name)
    detection_model = project.version(detection_project_version).model
 
    # load classification model
    classification_model.load_state_dict(torch.load(f=classification_model_path))

    if (cap.isOpened() == False): 
        print("Error opening video stream or file")

    iteration = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        iteration += 1
        if ret == True:
            image = Frame(frame, "./main/program_files/frame.jpg")
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
    main(video_path = './main/animation.gif', 
        detection_project_name ="dog-8syau", 
        detection_project_version = 2,
        classification_model_path = './digit_classification/train_model/model.pth',
        classification_model = MNISTModel(input_shape=3, hidden_units=10, output_shape=10))

