from dataset import Dataset
import cv2

def main():

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    path = "./digit_classification/create_dataset/data"
    
    train_data = dict(folder = "train", images_per_class = 700)
    valid_data = dict(folder = "valid", images_per_class = 200)
    test_data = dict(folder = "test", images_per_class = 100)
    folders = [train_data, valid_data, test_data]

    # create dataset
    dataset = Dataset()
    dataset.generate(classes, folders, path)

    # change fonts for digits 1, 6, 9
    fonts_for_1 = [cv2.FONT_HERSHEY_COMPLEX, 
                cv2.FONT_HERSHEY_TRIPLEX]
    dataset.modify_img_in_folder(folders, fonts_for_1, [1], path)

    fonts_for_6_9 = [cv2.FONT_HERSHEY_SIMPLEX,
                    cv2.FONT_HERSHEY_DUPLEX,
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    cv2.FONT_ITALIC]
    dataset.modify_img_in_folder(folders, fonts_for_6_9, [6, 9], path)


if __name__=="__main__":
    main()
