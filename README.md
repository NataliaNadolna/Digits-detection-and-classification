# Detection and classification digits on animation to calculate speed of the running dog

## Table of Contents

- [Introduction](#introduction)
- [Result](#result)
- [Dataset](#dataset)
- [Classification model](#classification-model)
- []()
- []()
- []()
- []()
- []()
- []()

  
## Introduction
This project is a tool for automatic analysis of animation. The program:
* detects a number indicating the distance covered by the dog,
* classifies this number,
* and calculates the speed of the dog's run based on it.

## Result
![](https://github.com/NataliaNadolna/Digits-detection-and-classification/blob/main/result.gif)

## Dataset
Created a dataset of images representing digits. Each image, sized 28x28 pixels, depicts a single digit (from 0 to 9). The color, font, size, thickness, and position of the digit on the image are specified in the Img_settings class.

### Image generation
To generate images, two classes were created: 
* the Writing_style class, which defines the style of the digit on the generated image;
* the Image class, which contains methods:
  - generate_empty() - to create a blank image,
  - write_text() - to write text on the image according to the specified style,
  - save() - to save the image.

### Dataset generation
Class Dataset has two methods:
* generate() - to generate dataset
* modify_img_in_folder() - to change font style in specific folders
  
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
* Linear Layer: A fully connected layer that takes the flattened features as input (in_features=hidden_units*7*7, as the spatial dimensions have been reduced by pooling operations) and produces an output tensor of size output_shape, which corresponds to the number of classes (10 classes) in a classification task.


## Training the model
### Parametres
The best results were achieved with the following parameters:
* 10 hidden units
* 5 epochs
* learning rate: 0.004
* batch size: 32
* optimizer: Adam

### Plots


### Confusion matrix
