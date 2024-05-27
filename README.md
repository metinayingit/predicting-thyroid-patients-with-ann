# 🧠 Predicting Thyroid Patients with Artificial Neural Networks (ANN)

This project aims to predict thyroid conditions using Artificial Neural Networks (ANN). The dataset consists of thyroid patient data with various features related to their health condition. The predictions categorize patients into one of three classes: Normal, Hyperthyroidism, and Hypothyroidism. This helps in determining whether a patient's thyroid gland is overactive, underactive, or functioning normally.

## 📊 Data Preparation

The dataset is divided into two parts: training and testing. Both datasets are preprocessed by scaling the features using StandardScaler to ensure that all features have the same scale.

## 🛠 Model Training

An MLPClassifier from the scikit-learn library is utilized to build the ANN model. The model architecture consists of three hidden layers, each containing 30 neurons. The model is trained on the training dataset with a maximum of 500 iterations.

## 📈 Model Evaluation

The trained model is evaluated using the testing dataset. Confusion matrix and classification report metrics are used to assess the performance of the model in predicting thyroid conditions. The predictions categorize the patients into one of three classes: Normal, Hyperthyroidism, and Hypothyroidism.

## 📋 Results

The confusion matrix provides insights into the model's performance in terms of true positive, true negative, false positive, and false negative predictions. The classification report summarizes the precision, recall, F1-score, and support for each class.

- **Normal**: The thyroid gland is functioning normally.
- **Hyperthyroidism**: The thyroid gland is overactive.
- **Hypothyroidism**: The thyroid gland is underactive.
