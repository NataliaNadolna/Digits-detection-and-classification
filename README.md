# Detection and classification digits on animation to calculate speed of the running dog
## Introduction
This project is a tool for automatic analysis of animation. The program:
* detects a number indicating the distance covered by the dog,
* classifies this number,
* and calculates the speed of the dog's run based on it.

## Result
![](https://github.com/NataliaNadolna/Digits-detection-and-classification/blob/main/result.gif)

## Dataset

## Detection

## Classification model
This model is a convolutional neural network (CNN) composed of multiple layers organized into three blocks: block_1, block_2, and classifier. Here's a breakdown of each block:

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

```python
summary(model=model_0,
        input_size=(32, 3, 28, 28), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
```
