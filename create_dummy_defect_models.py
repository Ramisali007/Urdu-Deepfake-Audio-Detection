import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron

# Define the MultiLabelDNN class
class MultiLabelDNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[256, 128]):
        super(MultiLabelDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[1], output_size)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Define the OnlinePerceptron class
class OnlinePerceptron:
    def __init__(self, n_features, n_labels):
        self.perceptrons = [Perceptron(alpha=0.01) for _ in range(n_labels)]
        self.n_features = n_features
        self.n_labels = n_labels
    
    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.n_labels))
        for i, perceptron in enumerate(self.perceptrons):
            y_pred[:, i] = np.random.randint(0, 2, size=X.shape[0])  # Dummy prediction
        return y_pred
    
    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0], self.n_labels))
        for i in range(self.n_labels):
            y_pred[:, i] = np.random.random(size=X.shape[0])  # Dummy probabilities
        return y_pred

def create_dummy_defect_models():
    print("Creating dummy defect prediction models...")
    
    # Define dimensions
    n_features = 100
    n_labels = 7
    
    # Create label column names
    label_columns = [
        'type_blocker',
        'type_regression',
        'type_bug',
        'type_documentation',
        'type_enhancement',
        'type_task',
        'type_dependency_upgrade'
    ]
    
    # Save label columns
    with open('defect_label_columns.txt', 'w') as f:
        for col in label_columns:
            f.write(f"{col}\n")
    
    # Create a dummy scaler
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(n_features)
    scaler.scale_ = np.ones(n_features)
    scaler.n_features_in_ = n_features
    joblib.dump(scaler, 'defect_feature_scaler.pkl')
    
    # Create dummy SVM model
    svm = MultiOutputClassifier(SVC(probability=True))
    # We need to set some attributes to make it work with predict and predict_proba
    svm.estimators_ = [SVC(probability=True) for _ in range(n_labels)]
    for est in svm.estimators_:
        est.classes_ = np.array([0, 1])
        est._n_features = n_features
    joblib.dump(svm, 'svm_defect_model.pkl')
    
    # Create dummy Logistic Regression model
    lr = MultiOutputClassifier(LogisticRegression())
    lr.estimators_ = [LogisticRegression() for _ in range(n_labels)]
    for est in lr.estimators_:
        est.classes_ = np.array([0, 1])
        est._n_features = n_features
    joblib.dump(lr, 'lr_defect_model.pkl')
    
    # Create dummy Perceptron model
    perceptron = MultiOutputClassifier(Perceptron())
    perceptron.estimators_ = [Perceptron() for _ in range(n_labels)]
    for est in perceptron.estimators_:
        est.classes_ = np.array([0, 1])
        est._n_features = n_features
    joblib.dump(perceptron, 'perceptron_defect_model.pkl')
    
    # Create dummy Online Perceptron model
    online_perceptron = OnlinePerceptron(n_features, n_labels)
    joblib.dump(online_perceptron, 'online_perceptron_defect_model.pkl')
    
    # Create dummy DNN model
    dnn_model = MultiLabelDNN(n_features, n_labels)
    torch.save(dnn_model.state_dict(), 'best_multilabel_dnn_model.pth')
    
    # Create dummy results DataFrame
    results = {
        'svm': {
            'hamming_loss': 0.15,
            'micro_f1': 0.85,
            'macro_f1': 0.82,
            'precision@1': 0.90
        },
        'logistic_regression': {
            'hamming_loss': 0.18,
            'micro_f1': 0.82,
            'macro_f1': 0.80,
            'precision@1': 0.88
        },
        'perceptron': {
            'hamming_loss': 0.20,
            'micro_f1': 0.80,
            'macro_f1': 0.78,
            'precision@1': 0.85
        },
        'online_perceptron': {
            'hamming_loss': 0.22,
            'micro_f1': 0.78,
            'macro_f1': 0.75,
            'precision@1': 0.82
        },
        'dnn': {
            'hamming_loss': 0.12,
            'micro_f1': 0.88,
            'macro_f1': 0.85,
            'precision@1': 0.92
        }
    }
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv('defect_prediction_results.csv')
    
    print("Dummy defect prediction models created successfully!")

if __name__ == "__main__":
    create_dummy_defect_models()
