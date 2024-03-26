# Detection and classification digits on animation to calculate speed of the running dog

## Table of Contents

- [Introduction](#introduction)
- [Result](#result)
- [Digits detection](#digits-detection)
- [Dataset for digits classification](#dataset-for-digits-classification)
- [Classification model](#classification-model)
- [Training the model](#training-the-model)
 
## Introduction
This project is a tool for automatic analysis of animation. The program:
* detects a number indicating the distance covered by the dog,
* classifies this number,
* calculates the speed of the dog's run based on it.

## Result
![](https://github.com/NataliaNadolna/Digits-detection-and-classification/blob/main/result.gif)

## Digits detection
The digit detection was performed using the Roboflow platform. Digits were annotated on 4 frames of the animation, and a model was trained for digit detection in the animation images.

![](https://github.com/NataliaNadolna/Digits-detection-and-classification/blob/main/prediction.jpg)

## Dataset for digits classification
Created a dataset of images representing digits. Each image, sized 28x28 pixels, depicts a single digit (from 0 to 9). The color, font, size, thickness, and position of the digit on the image are specified in the Img_settings class.
```python
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
```

### Image generation
To generate images, two classes were created: 
* the Writing_style class, which defines the style of the digit on the generated image. The colors, position and font scale are chosen randomly, while the fonts and thickness is selected sequentially.
* the Image class, which contains methods to:
  - create a blank image,
  - write text on the image according to the specified style,
  - save the image.

### Dataset generation
A dataset of 10,000 images has been created for 10 classes representing digits. It contains 900 training images and 100 test images per each class.
Class Dataset has two methods to:
* generate dataset;
* change font style in specific folders.

In created dataset folders with 1, 6 and 9 have changed fonts.
  
## Classification model
This model is a convolutional neural network (CNN) composed of multiple layers organized into three blocks: block_1, block_2, and classifier.
```python
========================================================================================================================
Layer (type (var_name))                  Input Shape          Output Shape         Param #              Trainable
========================================================================================================================
MNISTModel (MNISTModel)                  [32, 3, 28, 28]      [32, 10]             --                   True
├─Sequential (block_1)                   [32, 3, 28, 28]      [32, 10, 14, 14]     --                   True
│    └─Conv2d (0)                        [32, 3, 28, 28]      [32, 10, 28, 28]     280                  True
│    └─ReLU (1)                          [32, 10, 28, 28]     [32, 10, 28, 28]     --                   --
│    └─Conv2d (2)                        [32, 10, 28, 28]     [32, 10, 28, 28]     910                  True
│    └─ReLU (3)                          [32, 10, 28, 28]     [32, 10, 28, 28]     --                   --
│    └─MaxPool2d (4)                     [32, 10, 28, 28]     [32, 10, 14, 14]     --                   --
├─Sequential (block_2)                   [32, 10, 14, 14]     [32, 10, 7, 7]       --                   True
│    └─Conv2d (0)                        [32, 10, 14, 14]     [32, 10, 14, 14]     910                  True
│    └─ReLU (1)                          [32, 10, 14, 14]     [32, 10, 14, 14]     --                   --
│    └─Conv2d (2)                        [32, 10, 14, 14]     [32, 10, 14, 14]     910                  True
│    └─ReLU (3)                          [32, 10, 14, 14]     [32, 10, 14, 14]     --                   --
│    └─MaxPool2d (4)                     [32, 10, 14, 14]     [32, 10, 7, 7]       --                   --
├─Sequential (classifier)                [32, 10, 7, 7]       [32, 10]             --                   True
│    └─Flatten (0)                       [32, 10, 7, 7]       [32, 490]            --                   --
│    └─Linear (1)                        [32, 490]            [32, 10]             4,910                True
========================================================================================================================
Total params: 7,920
Trainable params: 7,920
Non-trainable params: 0
Total mult-adds (M): 41.43
========================================================================================================================
Input size (MB): 0.30
Forward/backward pass size (MB): 5.02
Params size (MB): 0.03
Estimated Total Size (MB): 5.35
========================================================================================================================
```

Here's a breakdown of each block:
### Block 1:
* Convolutional Layer 1: Applies a 2D convolution operation to the input image. It has input_shape channels as input and produces hidden_units channels as output. The convolutional kernel has a size of 3x3 and a padding of 1.
* ReLU Activation: Introduces non-linearity to the model by applying the rectified linear unit (ReLU) activation function element-wise.
* Convolutional Layer 2;
* Max Pooling Layer: Reduces the spatial dimensions of the feature maps by taking the maximum value within a 2x2 window and moving with a stride of 2. This downsampling operation helps in reducing computation and extracting dominant features.
### Block 2:
* Convolutional Layer 3;
* ReLU Activation;
* Convolutional Layer 4;
* ReLU Activation;
* Max Pooling Layer;
### Classifier:
* Flatten Layer: Flattens the feature maps into a 1D tensor, preparing them for input to the fully connected layers.
* Linear Layer: A fully connected layer that takes the flattened features as input (in_features=hidden_units x7x7, as the spatial dimensions have been reduced by pooling operations) and produces an output tensor of size output_shape, which corresponds to the number of classes (10 classes) in a classification task.

## Training the model
### Parametres
The best results were achieved with the following parameters:
* 10 hidden units
* 5 epochs
* learning rate: 0.004
* batch size: 32
* optimizer: Adam

### Plots
![](https://github.com/NataliaNadolna/Digits-detection-and-classification/blob/main/plots.png)

### Confusion matrix
![](https://github.com/NataliaNadolna/Digits-detection-and-classification/blob/main/matrix.png)
