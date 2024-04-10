Overview:

This repository contains the Jupyter Notebook and related resources for a project on emotion recognition using EEG (electroencephalogram) and facial data. The project applies machine learning and deep learning techniques to predict emotional states.

Dataset:

The dataset consists of EEG signals and facial video data from participants. EEG data are processed to extract relevant features, and facial data are normalized for use in the models.

Features:

Data preprocessing: EEG signal standardization and normalization.
Deep learning models: Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), specifically GRU and LSTM, to process the data.
Resampling techniques: SMOTE, Random Under/Oversampling for balancing the dataset.
Ensemble methods: A combination of XGBoost, Random Forest, and other classifiers are used.
Model evaluation: ROC curves, confusion matrices, and precision-recall metrics.

Prerequisites:

Python 3.x
Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow, keras, imbalanced-learn, xgboost, scikit-plot
Installation

Install the required Python libraries using:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras imbalanced-learn xgboost scikit-plot
Usage

Clone the repository.
Load and preprocess the dataset as per the instructions in the notebook.
Train the model using the data.
Evaluate the performance of the model using the provided metrics.
Experiment with different configurations and parameters.
Structure

Votting_classifier.ipynb: Main Jupyter Notebook containing the analysis and model training.
Data: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
