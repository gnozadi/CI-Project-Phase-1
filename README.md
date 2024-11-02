# EEG Signal Classification Project

## Overview
This project uses a machine learning pipeline to classify EEG (electroencephalogram) signals to detect seizure activity. The approach incorporates feature engineering, data normalization, and evaluating various classification models to optimize accuracy and performance.

## Objectives
- Implement a robust workflow for feature extraction from EEG data.
- Develop and compare the performance of different machine learning models, including SVM, Random Forest, and KNN.
- Employ k-fold cross-validation to evaluate model robustness and performance metrics such as accuracy, precision, and recall.

## Data and Preprocessing
The EEG data used in this project was preprocessed to ensure consistency and quality. Normalization was applied to input features to test its effect on classification accuracy, with results indicating minimal impact on models like Random Forest and SVM but significant improvements for KNN with higher k-values.

## Feature Engineering
A total of 15 unique features were extracted, including:
- **Statistical Features**: Mean, median, standard deviation, skewness, kurtosis, max, min, and variance.
- **Entropy Measures**: Shannon entropy, log energy entropy, Renyi entropy, and normalized entropy.
- **Time-Domain Features**: Peak-to-peak (PTP) and zero-crossing rate.
- **Innovative Features**: Negative sum (sum of negative signal values).

## Model Implementation
- **SVM**: Tested with different kernel types (Linear, Polynomial, RBF, Sigmoid). The Linear kernel provided the most consistent results.
- **Random Forest**: Optimal performance achieved with a max depth of 13, yielding high accuracy and precision.
- **KNN**: Performance varied with the number of neighbors; the best result was obtained with k = 6.



## Conclusion
The project successfully classified EEG signals with high accuracy, particularly using the Random Forest algorithm. Future work could involve exploring deep learning models for enhanced performance and applying the methods to real-time EEG monitoring applications.

## References
Link to relevant research or source material: UPF EEG Study
