import os
from image import Image
from img_settings import Img_settings
from writing import Writing_style

class Dataset():
    def generate(self, classes: list, dataset: list, data_path: str):
        os.mkdir(f"{data_path}")
        for data in dataset:
            os.mkdir(f"{data_path}/{data['folder']}")
            
            for c in classes:
                class_path = f"{data_path}/{data['folder']}/{c}"
                os.mkdir(class_path)

                for index in range(data['images_per_class']):
                    image = Image(Img_settings.img_size, 
                                  len(Img_settings.background_colors), 
                                  Img_settings.background_colors)
                    style = Writing_style(Img_settings.ink_colors, 
                                          Img_settings.number_position, 
                                          Img_settings.fonts, 
                                          Img_settings.font_scale, 
                                          Img_settings.thickness,
                                          index)
                    image.generate_empty()
                    image.write_text(style, c)

                    img_path = f"{data_path}/{data['folder']}/{c}/img{index}.png"
                    image.save(img_path) 

    def modify_img_in_folder(self, dataset: list, fonts: list, classes: list, data_path: str):
        for data in dataset:            
            for c in classes:
                for index in range(data['images_per_class']):
                    image = Image(Img_settings.img_size, 
                                  len(Img_settings.background_colors), 
                                  Img_settings.background_colors)
                    style = Writing_style(Img_settings.ink_colors, 
                                          Img_settings.number_position, 
                                          fonts, 
                                          Img_settings.font_scale, 
                                          Img_settings.thickness,
                                          index)
                    image.generate_empty()
                    image.write_text(style, str(c))

                    img_path = f"{data_path}/{data['folder']}/{c}/img{index}.png"
                    image.save(img_path)  

