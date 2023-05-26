# Facial emotion classification

## What is this project about?
In this project we made a machine learning classfication with several classic models. We tested all of them and found the best model, which was SVM. Models were training with PCA + LDA redusing dimenstions. We made a script that applies our result model to any image.

## Project content
There is a visual part of project in notebooks folder where were tested a bunch of classification models such as LDA, Desicion Tree, Random Forest, Ada boost, Gradient boosting, KNN and SVM. The main scripts for training and applying the best model are in src folder. Also there is a script that finds optinal parametrs for Gabor bank. 

## Dataset
In this project for emotion classification we used dataset contains of faces images with 7 classes: neutral (621 images), surprise (83), happiness (69), disgust (59), anger (45), fear (25) and contempt (18). Dataset is imbalanced, so we upsampled every class except neutral, but it still afected the result. Dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1YC9MjOC-qmR7eOE_oTXAL_03tUi3RN0r?usp=drive_link).
