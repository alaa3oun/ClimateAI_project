import torch
import torch.nn as nn
import numpy as np

# Define Early Stopping function
class EarlyStopping:
    def __init__(self, patience, threshold):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, loss):
        if self.best_loss - loss > self.threshold:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience 
    

# Define LSTM Model
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to generate future dates
def generate_future_dates(start_date, months):
    dates = []
    current_date = start_date
    for _ in range(months):
        # Handle month increment
        year = current_date.year + (current_date.month // 12)
        month = current_date.month % 12 + 1
        current_date = current_date.replace(year=year, month=month)
        dates.append(current_date)
    return dates
