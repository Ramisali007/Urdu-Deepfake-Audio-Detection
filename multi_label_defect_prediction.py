import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import hamming_loss, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    # Print dataset info
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    print("\nColumn information:")
    print(df.info())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values:\n", missing_values)

    # If there are missing values, fill them
    if missing_values.sum() > 0:
        print("Filling missing values...")
        # For numeric features, fill with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # For categorical features, fill with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Identify feature columns and target columns
    # We're assuming the last few columns are the defect labels
    # Adjust this based on your actual dataset structure

    # Example approach - modify as needed for your dataset
    # Assuming binary classification labels are at the end of the dataframe
    # and feature columns come before them
    # This is an assumption - you'll need to adapt this to your actual data structure

    # Check for columns that look like labels (binary values with 0 and 1)
    potential_label_cols = []
    for col in df.columns:
        unique_vals = df[col].unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
            potential_label_cols.append(col)

    # If we found potential label columns, use them as targets
    # Otherwise, assume the last quarter of columns are label columns
    if len(potential_label_cols) > 1:  # Assuming multiple label columns
        label_columns = potential_label_cols
        feature_columns = [col for col in df.columns if col not in label_columns]
    else:
        # If we couldn't identify label columns, guess
        n_cols = df.shape[1]
        feature_end_idx = int(0.75 * n_cols)  # Assuming last 25% are labels
        feature_columns = df.columns[:feature_end_idx]
        label_columns = df.columns[feature_end_idx:]

    print(f"\nFeature columns ({len(feature_columns)}):", feature_columns[:5], "...")
    print(f"Label columns ({len(label_columns)}):", label_columns)

    # Check label distribution
    print("\nLabel distribution:")
    for col in label_columns:
        print(f"{col}: {df[col].value_counts()}")

    # Check for imbalanced labels
    imbalance_ratios = []
    for col in label_columns:
        try:
            ratio = df[col].value_counts()[1] / df[col].value_counts()[0]
            imbalance_ratios.append((col, ratio))
        except:
            pass

    print("\nImbalance ratios (positive/negative):")
    for col, ratio in imbalance_ratios:
        print(f"{col}: {ratio:.4f}")

    # Process text features using TF-IDF
    print("\nProcessing text features...")
    text_features = df['report'].values

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)  # Limit to 100 features for simplicity
    X_text = vectorizer.fit_transform(text_features)

    # Convert sparse matrix to dense array for PyTorch compatibility
    X_text = X_text.toarray()

    # Save the vectorizer for later use
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    # Extract labels
    y = df[label_columns].values

    print(f"Features shape after TF-IDF: {X_text.shape}")
    print(f"Labels shape: {y.shape}")

    return X_text, y, label_columns

# Custom PyTorch Dataset for multi-label classification
class MultiLabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define DNN model for multi-label classification
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

# Train DNN model
def train_dnn_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device='cpu'):
    model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_hamming_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                # Convert to binary predictions
                preds = (outputs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_hamming = hamming_loss(np.array(val_labels), np.array(val_preds))

        # Save history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_hamming_loss'].append(val_hamming)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Hamming Loss: {val_hamming:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multilabel_dnn_model.pth')

    return history

# Evaluate model with multi-label metrics
def evaluate_multilabel_model(y_true, y_pred):
    metrics = {
        'hamming_loss': hamming_loss(y_true, y_pred),
        'micro_f1': f1_score(y_true, y_pred, average='micro'),
        'macro_f1': f1_score(y_true, y_pred, average='macro')
    }

    # Calculate precision@k
    def precision_at_k(y_true, y_pred, k):
        # For each sample, get the top k predicted labels
        n_samples = y_true.shape[0]
        total_precision = 0

        for i in range(n_samples):
            # Get indices of top k predictions
            top_k_indices = np.argsort(y_pred[i])[-k:]

            # Count how many of them are true positives
            true_positives = np.sum(y_true[i, top_k_indices])

            # Calculate precision@k for this sample
            precision = true_positives / k if k > 0 else 0
            total_precision += precision

        # Return average precision@k across all samples
        return total_precision / n_samples if n_samples > 0 else 0

    # Calculate precision@k for k=1, 3, 5 (if applicable)
    n_labels = y_true.shape[1]
    for k in [1, min(3, n_labels), min(5, n_labels)]:
        metrics[f'precision@{k}'] = precision_at_k(y_true, y_pred, k)

    return metrics

# Online Perceptron with partial_fit
class OnlinePerceptron:
    def __init__(self, n_features, n_labels):
        self.perceptrons = [Perceptron(alpha=0.01) for _ in range(n_labels)]
        self.n_features = n_features
        self.n_labels = n_labels

    def fit(self, X, y):
        # Initialize each perceptron with a single example
        for i, perceptron in enumerate(self.perceptrons):
            perceptron.partial_fit(X[:1], y[:1, i], classes=[0, 1])

        # Train each perceptron online
        for j in range(1, len(X)):
            self.partial_fit(X[j:j+1], y[j:j+1])

            if j % 100 == 0:
                print(f"Processed {j} samples")

        return self

    def partial_fit(self, X, y):
        for i, perceptron in enumerate(self.perceptrons):
            perceptron.partial_fit(X, y[:, i], classes=[0, 1])
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.n_labels))
        for i, perceptron in enumerate(self.perceptrons):
            y_pred[:, i] = perceptron.predict(X)
        return y_pred

    def predict_proba(self, X):
        # This is a rough approximation since Perceptron doesn't have predict_proba
        # We're using the decision function output and transforming to [0,1]
        y_pred = np.zeros((X.shape[0], self.n_labels))
        for i, perceptron in enumerate(self.perceptrons):
            try:
                # Get decision function values
                decisions = perceptron.decision_function(X)
                # Transform to pseudo-probabilities with sigmoid
                y_pred[:, i] = 1 / (1 + np.exp(-decisions))
            except:
                # Fallback to binary predictions
                y_pred[:, i] = perceptron.predict(X)
        return y_pred

# Plot training history
def plot_history(history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_hamming_loss'], label='Validation Hamming Loss')
    plt.legend()
    plt.title('Hamming Loss over epochs')

    plt.tight_layout()
    plt.savefig('multilabel_training_history.png')
    plt.close()

# Compare models
def compare_multilabel_models(results):
    models = list(results.keys())
    metrics = ['hamming_loss', 'micro_f1', 'macro_f1', 'precision@1']

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [results[model][metric] for model in models]
        plt.bar(models, values)
        plt.title(f'Comparison of {metric}')

        # For hamming loss, lower is better
        if metric == 'hamming_loss':
            plt.ylim(0, min(1.0, max(values) * 1.2))
        else:
            plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('multilabel_model_comparison.png')
    plt.close()

    # Also show as a table
    df_results = pd.DataFrame(results).T
    print(df_results)
    return df_results

def main():
    print("Starting multi-label defect prediction...")

    # Create synthetic data instead of loading from file
    print("Creating synthetic data for multi-label classification...")

    # Generate synthetic features and labels
    n_samples = 500
    n_features = 100
    n_labels = 7

    # Generate random features
    X = np.random.rand(n_samples, n_features)

    # Generate random binary labels
    y = np.random.randint(0, 2, size=(n_samples, n_labels))

    # Create label column names
    label_columns = [f'label_{i}' for i in range(n_labels)]

    print(f"Synthetic data created: {n_samples} samples with {n_features} features and {n_labels} labels")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    if X is None or y is None:
        print("Error: Dataset processing failed.")
        return None, None, None

    try:
        n_features = X.shape[1]
        n_labels = y.shape[1]

        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

        # Split data
        print("Splitting data into train/validation/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        # Standardize features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print("Feature standardization complete.")
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return None, None, None

    # Save the scaler
    joblib.dump(scaler, 'defect_feature_scaler.pkl')

    # Save label columns for reference
    with open('defect_label_columns.txt', 'w') as f:
        for col in label_columns:
            f.write(f"{col}\n")

    # Train traditional models
    results = {}

   # 1. Logistic Regression (One-vs-Rest)
    print("\n1. Training Logistic Regression (One-vs-Rest)...")
    lr = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results['logistic_regression'] = evaluate_multilabel_model(y_test, y_pred_lr)
    print("Logistic Regression metrics:", results['logistic_regression'])

    # Save the model
    joblib.dump(lr, 'lr_defect_model.pkl')

    # 2. SVM (One-vs-Rest)
    print("\n2. Training SVM (One-vs-Rest)...")
    svm = MultiOutputClassifier(SVC(probability=True))
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    results['svm'] = evaluate_multilabel_model(y_test, y_pred_svm)
    print("SVM metrics:", results['svm'])

    # Save the model
    joblib.dump(svm, 'svm_defect_model.pkl')

    # 3. Standard Perceptron (One-vs-Rest)
    print("\n3. Training Standard Perceptron (One-vs-Rest)...")
    perceptron = MultiOutputClassifier(Perceptron(max_iter=1000))
    perceptron.fit(X_train_scaled, y_train)
    y_pred_perceptron = perceptron.predict(X_test_scaled)
    results['perceptron'] = evaluate_multilabel_model(y_test, y_pred_perceptron)
    print("Perceptron metrics:", results['perceptron'])

    # Save the model
    joblib.dump(perceptron, 'perceptron_defect_model.pkl')

    # 4. Online Learning Perceptron (Challenge Element)
    print("\n4. Training Online Learning Perceptron...")
    online_perceptron = OnlinePerceptron(n_features, n_labels)
    online_perceptron.fit(X_train_scaled, y_train)
    y_pred_online = online_perceptron.predict(X_test_scaled)
    results['online_perceptron'] = evaluate_multilabel_model(y_test, y_pred_online)
    print("Online Perceptron metrics:", results['online_perceptron'])

    # Save the model
    joblib.dump(online_perceptron, 'online_perceptron_defect_model.pkl')

    # 5. Deep Neural Network
    print("\n5. Training Deep Neural Network...")

    # Create DataLoaders
    train_dataset = MultiLabelDataset(X_train_scaled, y_train)
    val_dataset = MultiLabelDataset(X_val_scaled, y_val)
    test_dataset = MultiLabelDataset(X_test_scaled, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize DNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dnn_model = MultiLabelDNN(n_features, n_labels)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label
    optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)

    # Train DNN
    history = train_dnn_model(dnn_model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device)

    # Plot training history
    plot_history(history)

    # Load best model
    dnn_model.load_state_dict(torch.load('best_multilabel_dnn_model.pth'))
    dnn_model.to(device)

    # Evaluate DNN on test set
    dnn_model.eval()
    y_pred_dnn = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = dnn_model(X_batch)
            preds = (outputs > 0.5).float().cpu().numpy()
            y_pred_dnn.extend(preds)

    y_pred_dnn = np.array(y_pred_dnn)
    results['dnn'] = evaluate_multilabel_model(y_test, y_pred_dnn)
    print("DNN metrics:", results['dnn'])

    # Compare all models
    print("\nComparing all models:")
    results_df = compare_multilabel_models(results)

    # Save results
    results_df.to_csv('defect_prediction_results.csv')

    print("All models trained and evaluated!")
    return results_df, n_features, n_labels  # Return dimensions for later use in the app

if __name__ == "__main__":
    main()