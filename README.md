# CoDaS - Indoor Localization

Indoor localization system using WiFi/RTT signal data.

## Dataset
RTT signal strength measurements (RSSI in dBm) collected across multiple locations in EETAC Auditorium using Google Pixel 3a devices.

## Pipeline
1. Load and preprocess RTT data
2. Feature selection using `SelectKBest` with `f_classif` (top 100 features)
3. Classification using `KNeighborsClassifier` (k=5)
4. Evaluation with Stratified K-Fold cross-validation (5 splits)

## Requirements
```bash
pip install pandas numpy scikit-learn
```

## Metrics
Each fold reports accuracy, precision, recall, and F1-score.
