import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.metrics import accuracy_score

def reshape_remove_characters(df):

    X = np.array([np.stack(row) for row in df.drop(columns=['Target']).values])
    y = df['Target'].values

    smote = SMOTE(random_state=42)
    n_samples, timesteps, n_features = X.shape
    X_flat = X.reshape((n_samples, timesteps * n_features))
    X_flat = np.where(X_flat == 'ç', 0, X_flat)

    X_resampled = X_flat.reshape((-1, timesteps, n_features))
    
    return X_resampled, y

def apply_smote(df, device):

    X = np.array([np.stack(row) for row in df.drop(columns=['Target']).values])
    y = df['Target'].values

    smote = SMOTE(random_state=42)
    n_samples, timesteps, n_features = X.shape
    X_flat = X.reshape((n_samples, timesteps * n_features))
    X_flat = np.where(X_flat == 'ç', 0, X_flat)

    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    X_resampled = X_resampled.reshape((-1, timesteps, n_features))
    
    return X_resampled, y_resampled

def convert_to_tensor(X_resampled, y_resampled, device, train_size, batch_size):
    # X_resampled = X_resampled.squeeze(axis=1) 
    
    if train_size < 1:
        pct_test_size = 1- train_size
        
    else:
        pct_test_size = 1 - train_size / X_resampled.shape[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, pct_test_size, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return trainloader, testloader
    
class GRU3DClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(GRU3DClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Only use if suitable for your classification type
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate through GRU
        out, _ = self.gru(x, h0)
        # Take output from the last time step
        out = self.fc(out[:, -1, :]) 
        return self.softmax(out)  # Consider using this only if it's suitable for your classification
    
def train_model(model, optimizer, num_epochs, trainloader, criterion, device):

    losses = []
    predictions_list = []

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        epoch_predictions = []
        epoch_realized = []
        for X_batch, y_batch in trainloader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            optimizer.zero_grad()

            pred_y = model(X_batch)
            loss = criterion(pred_y, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            # Detach predictions and convert to CPU for analysis (if using GPU)
            epoch_predictions.append(pred_y.detach().cpu().numpy())
            epoch_realized.append(y_batch.detach().cpu().numpy())

        # Calculate the average loss for the epoch
        average_loss = running_loss / len(trainloader)

        # Store the average loss for this epoch
        losses.append(average_loss)
        epoch_predictions = np.concatenate(epoch_predictions, axis=0)
        epoch_realized = np.concatenate(epoch_realized, axis=0)
        predictions_list.append(epoch_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}")

    # After training, plot the loss
    plt.plot(range(1, num_epochs+1), losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    predicted_y = np.argmax(predictions_list[-1], axis=1)  # Shape: (1000,)
    proportion_pred_over_0_5 = np.mean(predicted_y)
    proportion_realised_ones = np.mean(np.array(epoch_realized) == 1)  # Ensure epoch_realized is an array

    print(f"Proportion of Predicted 1's: {proportion_pred_over_0_5:.2f}\n"
        f"Proportion of Realized 1's: {proportion_realised_ones:.2f}")
    
    conf_matrix = confusion_matrix(epoch_realized, predicted_y)

    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Last Epoch)')
    plt.show()

    TN, FP, FN, TP = conf_matrix.ravel()  # Unravel the confusion matrix
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Print accuracy
    print(f"Accuracy: {accuracy:.2f}")
    
    return accuracy


def evaluate_model(model, testloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():  # Disable gradient calculation
        for X_batch, y_batch in testloader:
    
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            
            
            pred_y = model(X_batch)
            
            loss = criterion(pred_y, y_batch)
            total_loss += loss.item()
            
            all_predictions.append(pred_y.detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())
    
    # Calculate average loss
    average_loss = total_loss / len(testloader)
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    predicted_classes = np.argmax(all_predictions, axis=1)  # Assuming softmax is used and you want class labels
    accuracy = accuracy_score(all_targets, predicted_classes)
    
    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy


def evaluate_rolling_model(model, X, y, criterion, optimizer, device, train_size, train_batch_size, test_batch_size, num_epochs):
    """
    Evaluate a PyTorch model using a rolling prediction approach for time series.
    
    Args:
        model: PyTorch model to evaluate.
        X: Resampled feature data (numpy array).
        y: Resampled target data (numpy array).
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer (e.g., Adam).
        device: Device for computation (CPU or GPU).
        train_size: Initial size of the training data.
        train_batch_size: Batch size for training.
        test_batch_size: Batch size for testing.
        num_epochs: Number of epochs for training at each step.
    
    Returns:
        dict: Dictionary containing rolling predictions, targets, and metrics.
    """
    rolling_predictions = []
    rolling_targets = []
    rolling_losses = []
    
    if train_size < 1: # If its in percentage terms:
        lower_bound = int(train_size*len(X))
    else:
        lower_bound = train_size

    # Loop through the test set incrementally
    for i in range(lower_bound, len(X)):
        print(f"Processing step {i}/{len(X)}...")
        
        # Prepare the rolling training and test sets
        X_train_data = X[:i]
        X_test_data = X[i:i+1]
        y_train_data = y[:i]
        y_test_data = y[i:i+1]
        
        X_train = torch.tensor(X_train_data, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test_data, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train_data, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test_data, dtype=torch.long).to(device)
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        
        # Training
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in trainloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                pred_y = model(X_batch)
                loss = criterion(pred_y, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

        # Append training loss for this step
        rolling_losses.append(epoch_loss / len(trainloader))

        # Evaluation
        model.eval()
        step_predictions = []
        step_targets = []

        with torch.no_grad():
            for X_batch, y_batch in testloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                pred_y = model(X_batch)
                step_predictions.append(pred_y.detach().cpu().numpy())
                step_targets.append(y_batch.detach().cpu().numpy())

        # Concatenate predictions and targets for this step
        step_predictions = np.concatenate(step_predictions, axis=0)
        predicted_classes = np.argmax(step_predictions, axis=1)  # Convert logits to class labels
        step_targets = np.concatenate(step_targets, axis=0)

        rolling_predictions.append(predicted_classes)
        rolling_targets.append(step_targets)

    # Flatten the rolling predictions and targets
    rolling_predictions = np.concatenate(rolling_predictions, axis=0)
    rolling_targets = np.concatenate(rolling_targets, axis=0)
    
    # Calculate metrics
    test_accuracy = accuracy_score(rolling_targets, rolling_predictions)
    print(f"Rolling Test Accuracy: {test_accuracy:.4f}")

    return rolling_predictions, rolling_targets, test_accuracy
