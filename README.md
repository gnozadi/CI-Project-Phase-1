# EEG Signal Classification Project

## Overview
This project aims to develop and evaluate a system for detecting seizures using EEG (electroencephalogram) data. The focus is on processing EEG signals to classify seizure and non-seizure events using a combination of feature engineering and machine learning models.

## Objectives
- Implement a robust pipeline for loading and preprocessing EEG data.
- Develop feature extraction techniques using time and frequency domain features and LBP-based characteristics.
- Evaluate machine learning models for classification, including SVM, Random Forest, and KNN.
- Employ k-fold cross-validation for reliable performance assessment.

## Data Source
The EEG dataset used in this project was gathered with a specific focus on reliable and standardized signal acquisition, ensuring high-quality data for training and testing purposes.

## Methodology
1. **Data Preprocessing**: 
   - Loading raw EEG data and preparing it for analysis by segmenting the signals into manageable sequences.
2. **Feature Extraction**:
   - Employing statistical, time-domain, frequency-domain, and LBP-based features to extract meaningful information from EEG signals.
3. **Model Training**:
   - Implementing and fine-tuning SVM, Random Forest, and KNN classifiers.
   - Cross-validation with a k-fold approach to ensure the robustness of the evaluation.
4. **Performance Metrics**:
   - Reporting accuracy, precision, recall, and providing confusion matrices for a comprehensive evaluation.

## Results
The models' performances were evaluated based on precision, recall, and overall accuracy, with detailed analysis comparing their strengths in different contexts.

## Conclusion
This project demonstrates the capability to automate the detection of seizure events using EEG data with a solid foundation in machine learning, providing valuable insights for potential applications in healthcare.

## References
- Link to relevant research or source material: [UPF EEG Study](https://www.upf.edu/web/ntsa/downloads/-/asset_publisher/xvT6E4pczrBw/content/2001-indications-of-nonlinear-deterministic-and-finite-dimensional-structures-in-time-series-of-brain-electrical-activity-dependence-on-recording-regi)

