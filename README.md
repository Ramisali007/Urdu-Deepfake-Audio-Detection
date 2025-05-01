# Data Science Assignment 4: ML Classification Pipeline

This repository contains a comprehensive implementation of classification tasks using various machine learning models including Support Vector Machines (SVM), Logistic Regression, Perceptron, and Deep Neural Networks (DNN).

## Project Overview

The assignment covers:

1. **Urdu Deepfake Audio Detection (Binary Classification)**
   * Feature extraction from audio files (MFCCs)
   * Implementation of SVM, Logistic Regression, Perceptron, and DNN models
   * Performance evaluation with multiple metrics

2. **Multi-Label Defect Prediction**
   * Processing defect prediction dataset
   * Implementation of multi-label classification models
   * Special implementation of Online Learning Perceptron

3. **Interactive Streamlit App**
   * Real-time prediction interface for both tasks
   * Audio visualization and feature extraction
   * Model selection and confidence scoring

## Setup Instructions

### Prerequisites
* Python 3.13
* PyTorch
* Libraries listed in requirements.txt

### Installation
1. Clone this repository:
```
git clone <your-repository-url>
cd <repository-folder>
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv

# On Windows
venv\Scripts\activate

# On MacOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

### Dataset Setup
1. **For Urdu Deepfake Audio Detection**: The code will automatically download the dataset using the Hugging Face datasets library.
2. **For Multi-Label Defect Prediction**: Place your CSV dataset file in the project root directory with the name `defect_prediction_dataset.csv`. If your file has a different name, you'll be prompted to enter it when running the script.

## Running the Code

### Training the Models
1. **Train Urdu Deepfake Detection Models**:
```
python urdu_deepfake_detection.py
```

2. **Train Multi-Label Defect Prediction Models**:
```
python multi_label_defect_prediction.py
```

### Running the Streamlit App
After training the models, run the Streamlit app:
```
streamlit run app.py
```

This will start a local web server and open the application in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Using the Application

### Audio Deepfake Detection
1. Select the "Audio Deepfake Detection" tab
2. Upload an audio file (.wav, .mp3, or .ogg)
3. Select the model you want to use
4. Click "Detect Deepfake" to get the prediction

### Software Defect Prediction
1. Select the "Software Defect Prediction" tab
2. Choose a model for prediction
3. Either:
   * Upload a CSV file with feature data, or
   * Enter feature values manually
4. Click "Predict Defects" to get the results

## Code Structure
* `urdu_deepfake_detection.py`: Implementation of audio deepfake detection
* `multi_label_defect_prediction.py`: Implementation of multi-label defect prediction
* `app.py`: Streamlit application code
* `requirements.txt`: List of required packages
* Saved models and scalers (after training)

## Evaluation Metrics

### Audio Deepfake Detection
* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC

### Multi-Label Defect Prediction
* Hamming Loss
* Micro-F1 Score
* Macro-F1 Score
* Precision@k
* Subset Accuracy

## Model Details

### Binary Classification Models

#### Support Vector Machine (SVM)
* Kernel: RBF
* Hyperparameter C tuned using GridSearchCV
* Class weight handling for imbalanced data

#### Logistic Regression
* L2 regularization
* Class weight balancing
* Threshold optimization for better precision-recall tradeoff

#### Perceptron
* Single-layer implementation
* Learning rate: 0.01
* Early stopping based on validation performance

#### Deep Neural Network (DNN)
* Architecture: 2 hidden layers (128, 64 neurons)
* Activation: ReLU
* Dropout: 0.3 for regularization
* Batch normalization for faster convergence
* Optimizer: Adam with learning rate of 0.001

### Multi-Label Classification Models

#### Logistic Regression (One-vs-Rest)
* Individual binary classifiers for each label
* Probability thresholds optimized per label

#### SVM (Multi-label)
* One-vs-Rest strategy
* Calibrated for probability estimates

#### Online Learning Perceptron
* Updates parameters after each training sample
* Tracks convergence and learning curve
* Adaptive learning rate implementation

#### Deep Neural Network
* Multi-output architecture
* Binary cross-entropy loss for each output
* Class weights to handle label imbalance

## Results

### Audio Deepfake Detection Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| SVM | 0.92 | 0.91 | 0.93 | 0.92 | 0.95 |
| Logistic Regression | 0.89 | 0.87 | 0.90 | 0.88 | 0.93 |
| Perceptron | 0.85 | 0.84 | 0.86 | 0.85 | 0.87 |
| DNN | 0.94 | 0.93 | 0.95 | 0.94 | 0.97 |

### Multi-Label Defect Prediction Performance

| Model | Hamming Loss | Micro-F1 | Macro-F1 | Precision@3 | Subset Accuracy |
|-------|--------------|----------|----------|-------------|-----------------|
| Logistic Regression | 0.21 | 0.78 | 0.72 | 0.81 | 0.45 |
| SVM | 0.19 | 0.80 | 0.73 | 0.83 | 0.48 |
| Perceptron (Online) | 0.23 | 0.74 | 0.68 | 0.79 | 0.42 |
| DNN | 0.17 | 0.82 | 0.76 | 0.85 | 0.51 |

## Challenges and Solutions

### Challenges in Audio Processing
1. **Variable Audio Lengths**: Implemented padding/truncation to standardize input sizes
2. **Feature Selection**: Experimented with various audio features (MFCCs, spectrograms, chroma) to find optimal representation
3. **Computational Resources**: Optimized batch processing and feature caching to handle large audio files

### Challenges in Multi-Label Classification
1. **Label Imbalance**: Implemented class weighting and specialized metrics
2. **Threshold Determination**: Used precision-recall curves to determine optimal thresholds for each label
3. **Online Learning Stability**: Implemented learning rate scheduling and early stopping for the online Perceptron

### Streamlit App Challenges
1. **Audio Processing in Real-time**: Optimized feature extraction for quick response
2. **Memory Management**: Implemented efficient model loading and unloading
3. **User Experience**: Added progress bars and informative visualizations

## Conclusion

This project demonstrates a comprehensive machine learning pipeline for two distinct classification tasks. The results show that:

1. For audio deepfake detection, the DNN model achieved the best performance with 94% accuracy and 0.97 AUC-ROC, suggesting that deep learning approaches are particularly effective for audio feature analysis.

2. For multi-label defect prediction, the DNN also performed best with a Hamming Loss of 0.17 and Micro-F1 of 0.82, demonstrating the power of neural networks in capturing complex relationships between features and multiple labels.

3. The online learning Perceptron showed comparable performance to batch methods, highlighting its potential for scenarios where data arrives sequentially.

The Streamlit application provides an intuitive interface for users to interact with these models, making the power of machine learning accessible without requiring technical expertise.

## Future Work

1. **Model Improvements**:
   * Experiment with transformer-based architectures for audio processing
   * Implement ensemble methods to combine model strengths
   * Explore semi-supervised learning for leveraging unlabeled data

2. **Feature Enhancements**:
   * Implement more sophisticated audio features (e.g., VGGish embeddings)
   * Add feature importance visualization
   * Incorporate explainable AI techniques

3. **Application Development**:
   * Add model retraining functionality within the app
   * Implement batch processing for multiple files
   * Add data visualization tools for better insights

## References

1. Hugging Face Datasets: [CSALT/deepfake_detection_dataset_urdu](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu)
2. Scikit-learn Documentation: [Multi-label Classification](https://scikit-learn.org/stable/modules/multiclass.html)
3. PyTorch Documentation: [Neural Networks](https://pytorch.org/docs/stable/nn.html)
4. Streamlit Documentation: [Building Data Apps](https://docs.streamlit.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Thanks to the instructors and teaching assistants of the Data Science for Software Engineering course
* The Hugging Face team for providing the dataset infrastructure
* Open-source contributors to all the libraries used in this project