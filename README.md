## Fitness Exercise Classifier Model 
## Overview

This repository contains a deep learning model for classifying fitness exercises based on Frames extracted from videos. The model is built using TensorFlow and Opencv and Mediapipe and is capable of recognizing various exercises such as Lunges, Side Plank, and Upper Crunches and evaluates it as true or wrong.

<br>

## Installation

### Clone the repository

git clone [AI-Gym-Coach](https://github.com/ShimaaAbdalraheem/AI-Gym-Coach-)
<br>

1. Navigate to the project directory:

2. Install dependencies

pip install -r requirements.txt


## Model Architecture

<br>
The model architecture is based on a convolutional neural network (CNN) implemented using TensorFlow's Keras API. It consists of multiple convolutional layers followed by fully connected layers, ending with a softmax activation function to output class probabilities.

<br>

## Dataset
<br>
The model was trained on a custom dataset consisting of images that captured from different videos of individuals performing different fitness exercises. These images were annotated with pose landmarks using the MediaPipe Pose Detection framework.

<br>
<br>

## IMAGE COLLECTING:
 <br>
-to collect images: 
1- Collect Videos of people Performing the exercises you want
<br>
2- run 'DataCollection.py' to start extracting Frames of the videos

<br> 

## DATA PROCESSING:

- after collecting all the Videos and running DataCollection.py to extract frames ..
, Run the file 'DataPreprocessing.py' to extract poe landmarks from each frame and store them .

## MODEL TRAINING:
<br>
- after doing data prossesing go to Model.py file or Model.ipynb(same) to train mode.
- modify NUM_CLASSES and exercise_labels to.
- run the file.

## Model testing:
- for testing model via web cam run AITrainer.py file. 
<br>

## Files
<br>


DataCollection.py: this script is used to extract frames from videos and storing them is classes
<br>
DataPreprocessing.py :  this script is used to extract pose Landmarks from the Photos and storing them in Numpy array
<br>

Model.py: This script is used for training the model using the provided dataset. It loads the dataset, preprocesses the data, defines and compiles the model architecture, trains the model, and saves the trained model to a specified location.
also This script is for evaluating the model's performance and generating analysis reports. It loads the trained model, evaluates it on a separate test dataset, calculates evaluation metrics such as accuracy, precision, recall, and F1-score, and generates a confusion matrix for visualizing the model's performance across different classes.

<br>
AiTrainer.py: This script allows for real-time inference using the trained model on webcam or video input. It loads the trained model, processes input frames using MediaPipe Pose Detection, and outputs predictions for each frame.

<br>
Model.ipynb: This script is the same file as Model.py.
<br>
results folder: this folder contains result analysis of the Model such as confusion matrix.
<br>

requirements.txt: This file contains a list of dependencies required to run the scripts. You can install these dependencies using pip install -r requirements.txt.
<br>



## EVALUATION

-run the file 'Model.py' to evalute model performance and calculate accuracy ,and to produce the confusion_matrix
<br>
