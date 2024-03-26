from dataset import Dataset
import cv2

train_data = dict(folder = "train", images_per_class = 900)
test_data = dict(folder = "test", images_per_class = 100)
folders = [train_data, test_data]

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# create dataset
dataset = Dataset()
dataset.generate(folders, classes)

# change fonts in folders 1, 6, 9
fonts_for_1 = [cv2.FONT_HERSHEY_COMPLEX, 
               cv2.FONT_HERSHEY_TRIPLEX]
dataset.modify_img_in_folder(folders, fonts_for_1, [1])

fonts_for_6_9 = [cv2.FONT_HERSHEY_SIMPLEX,
                 cv2.FONT_HERSHEY_DUPLEX,
                 cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                 cv2.FONT_ITALIC]
dataset.modify_img_in_folder(folders, fonts_for_6_9, [6, 9])
