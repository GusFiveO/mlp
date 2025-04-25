# Breast Cancer Prediction with Multilayer Perceptron

## Overview

This project implements a Multilayer Perceptron (MLP) from scratch to predict breast cancer diagnoses. The MLP is trained on the Breast Cancer Wisconsin dataset, which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. The goal is to classify the tumors as malignant or benign based on these features.

## Table of Contents

1. [Technical Aspects](#technical-aspects)
   - [Dataset](#dataset)
   - [Model Architecture](#model-architecture)
   - [Training Process](#training-process)
   - [Evaluation Metrics](#evaluation-metrics)
2. [Usage](#usage)
   - [Installation](#installation)
   - [Training the Model](#training-the-model)
   - [Making Predictions](#making-predictions)
3. [Results](#results)
   - [Output](#output)
   - [Training and Validation Metrics](#training-and-validation-metrics)
   - [Confusion Matrix](#confusion-matrix)
4. [Visualizations](#visualizations)
   - [Correlation Heatmaps](#correlation-heatmaps)

## Technical Aspects

### Dataset

The Breast Cancer Wisconsin dataset is used for training and evaluating the MLP. The dataset contains 569 samples, each with 30 numeric features computed from digitized images of FNA of breast masses. The features describe characteristics of the cell nuclei present in the image. The target variable is the diagnosis, which is either malignant (M) or benign (B).

### Model Architecture

The MLP is implemented using the following components:

- **Layer Class**: Defines a single layer in the neural network, including initialization of weights and biases, forward and backward propagation, and update rules.
- **NeuralNetwork Class**: Manages the layers, training process, and evaluation metrics. It includes methods for initializing layers, forward and backward propagation, gradient descent, and early stopping.
- **Activation Functions**: Sigmoid and Softmax activation functions are used in the hidden and output layers, respectively.
- **Weight Initialization**: Weights are initialized using uniform distribution or Xavier initialization.

### Training Process

The training process involves the following steps:

1. **Data Preprocessing**: The features are normalized, and the targets are one-hot encoded.
2. **Layer Initialization**: The layers are initialized based on the specified architecture and weight initialization method.
3. **Forward Propagation**: The input data is passed through the layers to compute the output.
4. **Backward Propagation**: The error is computed, and gradients are propagated back through the network to update the weights.
5. **Gradient Descent**: The weights are updated using gradient descent with an optional momentum term.
6. **Early Stopping**: Training is stopped early if the validation loss does not improve for a specified number of epochs.

### Evaluation Metrics

The model's performance is evaluated using the following metrics:

- **Binary Cross Entropy Loss**: Measures the difference between the predicted and true distributions.
- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A summary of prediction results for classification problems.

## Usage

### Installation

To run the project, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python multi_layer_preceptron.py -train --learning_rate 0.03 --epochs 2500 --shape 10 10 --momentum 0.9 data/breast_cancer.csv
```

### Making Predictions

To make predictions using the trained model, use the following command:

```bash
python multi_layer_preceptron.py -predict data/breast_cancer.csv
```

## Results

### Output

![Program output](images/MLP-metrics-output.png)

### Training and Validation Metrics

![Training and Validation Metrics](images/MLP-training.png)

### Confusion Matrix

![Confusion Matrix](images/MLP-confusion-matrix.png)

## Visualizations

### Correlation Heatmaps

The `visualization.py` script generates correlation heatmaps to visualize the relationships between features in the dataset.

#### Before feature selection

![Correlation Heatmap](images/MLP-correlation-heatmap.png)

#### After feature selection

![Correlation Heatmap](images/MLP-feature-selection-heatmap.png)
