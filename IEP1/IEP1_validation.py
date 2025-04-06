import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from functions import EarlyStopping, MultiOutputLSTM, create_sequences

# Load the cleaned dataset
file_path = "cleaned_data.csv" 
df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
selected_features = ["aet", "def", "pdsi", "pet", "soil", "srad", "tmmn", "tmmx", "vap", "vpd", "vs"]
df_selected = df[["date"] + selected_features].copy()

# Train (1975-1999) and Validation (2000-2024) Split
train_df = df_selected[(df_selected['date'] >= '1975-01-01') & (df_selected['date'] <= '1999-12-01')]
val_df = df_selected[(df_selected['date'] >= '2000-01-01') & (df_selected['date'] <= '2024-12-01')]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[selected_features])
val_scaled = scaler.transform(val_df[selected_features])

# Create sequences
seq_length = 60  # 5 years
X_train, y_train = create_sequences(train_scaled, seq_length)
X_val, y_val = create_sequences(val_scaled, seq_length)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

# Model parameters
input_size = len(selected_features)
hidden_size = 64
num_layers = 2
output_size = len(selected_features)

model = MultiOutputLSTM(input_size, hidden_size, num_layers, output_size)

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
Early_Stopping = EarlyStopping(patience=300, threshold=0.1)  
# Training and Validation Loop
def train_validate_model(model, x_train, y_train, x_val, y_val, criterion, optimizer, epochs=2000):

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        train_loss = criterion(output, y_train)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_output = model(x_val)
            val_loss = criterion(val_output, y_val).item()

        if Early_Stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break  

        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss:.4f}")

    val_predictions = scaler.inverse_transform(np.array(val_output))
    val_actual = scaler.inverse_transform(y_val.numpy())
    val_start_date = val_df["date"].iloc[seq_length]  # the first prediction corresponds to this date
    val_years = pd.date_range(val_start_date, periods=len(val_predictions), freq='M')


    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "validation_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Plot validation predictions vs. actual
    for i, feature in enumerate(selected_features):
        plt.figure(figsize=(12, 6))
        plt.plot(val_years, val_actual[:, i], label="Actual")
        plt.plot(val_years, val_predictions[:, i], label="Predicted", linestyle='dashed')
        plt.xlabel("Year")
        plt.ylabel(feature)
        plt.title(f"Validation: {feature} (2000-2025)")
        plt.legend()
        
        # Save the plot
        plot_filename = f"{feature}_validation_plot.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

    

# Train and validate the model
train_validate_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, epochs=2000)