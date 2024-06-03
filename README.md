# CIFAR-10 Convolutional Neural Network

## Introduction to Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a class of deep learning algorithms that are particularly effective for analyzing visual data. Unlike traditional neural networks, CNNs use a specialized architecture that is designed to take advantage of the 2D structure of input data, such as images.

### Key Features of CNNs:

- **Convolutional Layers**: These layers apply a series of filters (kernels) to the input image to create feature maps. Each filter detects different features such as edges, textures, and patterns.
- **Pooling Layers**: These layers reduce the spatial dimensions of the feature maps, which helps to reduce the number of parameters and computational cost. Pooling also helps in making the detection of features invariant to scale and orientation changes.
- **Fully Connected Layers**: After several convolutional and pooling layers, the network may include fully connected layers that are similar to those in traditional neural networks. These layers perform the final classification based on the extracted features.

### Neural Network Architecture Visualization

This section includes code to visualize the neural network architecture. The diagram helps to understand the structure and connectivity of different layers in the network.

<img src="https://github.com/joemcmullin/Convolutional-CIFAR_10-Neural-Network/assets/3474363/9d6a9361-a9d8-4000-8b5b-440f85e4e3c4" alt="Description" width="50%"/>

### CNNs vs. Traditional Neural Networks:

- **Spatial Hierarchy**: CNNs maintain the spatial hierarchy of the image by using local connections and pooling, whereas traditional neural networks treat the input data as a flat vector.
- **Parameter Sharing**: CNNs use parameter sharing (same weights for different parts of the input) in convolutional layers, reducing the number of parameters and improving generalization.
- **Better Performance on Visual Data**: Due to their specialized architecture, CNNs significantly outperform traditional neural networks on tasks involving visual data, such as image classification, object detection, and image segmentation.

## Project Overview

This repository contains code to build, train, and visualize a Convolutional Neural Network (CNN) using the CIFAR-10 dataset. The project covers data loading, exploration, CNN model definition, training, evaluation, and visualization.

### Viewing Combined Dataset Summary
Create and display a combined summary of the dataset to understand the distribution of classes in both training and test sets.

<img src="https://github.com/joemcmullin/Convolutional-CIFAR_10-Neural-Network/assets/3474363/e1d24560-6a75-453e-96ed-3b1ee1adbf1f" alt="Description" width="50%"/>


### CNN Model Definition
Define the architecture of the Convolutional Neural Network (CNN). The model includes multiple convolutional layers to extract features from images, pooling layers to reduce dimensionality, and dense layers for classification.

<img src="https://github.com/joemcmullin/Convolutional-CIFAR_10-Neural-Network/assets/3474363/70cc1029-b19b-4275-a377-ab534d1a4e50" alt="Description" width="50%"/>


### Training Process
Create an animation to visualize the training process. This animation shows how the accuracy changes over epochs for both training and validation sets.

<img src="https://github.com/joemcmullin/Convolutional-CIFAR_10-Neural-Network/assets/3474363/3b6cb758-ed83-4a72-925d-9c5b06a5c9f6" alt="Description" width="50%"/>


### Confusion Matrix
Compute and visualize the confusion matrix to evaluate the model's performance. The confusion matrix provides insights into the types of errors the model is making.

<img src="https://github.com/joemcmullin/Convolutional-CIFAR_10-Neural-Network/assets/3474363/184780ba-3fc4-43ba-a384-e6c3d41d805c" alt="Description" width="50%"/>


## Requirements

Before running the code using the source Juypter Notebook file in the code, ensure you have the following packages installed:

- TensorFlow
- Keras
- Matplotlib
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Graphviz
- IPython
