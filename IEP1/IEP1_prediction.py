import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
from functions import EarlyStopping, MultiOutputLSTM, create_sequences, generate_future_dates

# Load the cleaned dataset
file_path = "cleaned_data.csv" 
df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
selected_features = ["aet", "def", "pdsi", "pet", "soil", "srad", "tmmn", "tmmx", "vap", "vpd", "vs"]
df_selected = df[["date"] + selected_features].copy()

# Train (1975-1999) and Validation (2000-2024) Split
train_df = df_selected[(df_selected['date'] >= '1975-01-01') & (df_selected['date'] <= '2024-12-01')]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[selected_features])

# Create sequences
seq_length = 60  # 5 years
X_train, y_train = create_sequences(train_scaled, seq_length)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)

# Model parameters
input_size = len(selected_features)
hidden_size = 64
num_layers = 2
output_size = len(selected_features)

model = MultiOutputLSTM(input_size, hidden_size, num_layers, output_size)

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training and Validation Loop
def train_model(model, x_train, y_train, criterion, optimizer, epochs=2000):

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        train_loss = criterion(output, y_train)
        train_loss.backward()
        optimizer.step()

        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}")


# Predict future time steps given a user-defined end date
def predict_future(model, train_df, scaler, selected_features, seq_length, target_end_date_str):
    model.eval()

    # Get the last sequence from the training data
    recent_data = train_df[selected_features].values[-seq_length:]
    input_seq = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len, features)

    # Parse the target end date
    target_end_date = pd.to_datetime(target_end_date_str)
    last_known_date = train_df["date"].iloc[-1]
    
    # Calculate number of months to predict
    months_to_predict = (target_end_date.year - last_known_date.year) * 12 + (target_end_date.month - last_known_date.month)
    if months_to_predict <= 0:
        raise ValueError("Target date must be after the last date in the training set.")

    predictions = []
    future_dates = []

    for _ in range(months_to_predict):
        with torch.no_grad():
            next_pred = model(input_seq).squeeze(0).numpy()  # Predict next step
        predictions.append(next_pred)
        future_dates.append(last_known_date + pd.DateOffset(months=len(future_dates)+1))

        # Update input sequence
        next_input = np.vstack([input_seq.squeeze(0).numpy()[1:], next_pred])
        input_seq = torch.tensor(next_input, dtype=torch.float32).unsqueeze(0)

    # Inverse transform to original scale
    predictions = scaler.inverse_transform(predictions)
    prediction_df = pd.DataFrame(predictions, columns=selected_features)
    prediction_df["date"] = future_dates

    return prediction_df


train_model(model, X_train, y_train, criterion, optimizer, epochs=500)

# Example usage
user_input_date = "2030-01-01"  # or input("Enter prediction end date (YYYY-MM-DD): ")
predicted_df = predict_future(model, train_df, scaler, selected_features, seq_length, user_input_date)

# Save the predicted DataFrame
predicted_df.to_csv("predicted_future_data.csv", index=False)
print(predicted_df.head())

# Create an output directory if it doesn't exist
output_dir = "prediction_plots"
os.makedirs(output_dir, exist_ok=True)

# Plot and save each feature over time
for feature in selected_features:
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_df["date"], predicted_df[feature], label=f"Predicted {feature}", linestyle='dashed', color='tab:blue')
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.title(f"Prediction: {feature} (2025-2030)")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_filename = f"{feature}_prediction_plot.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

