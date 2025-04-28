import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from CarDataset import CarDataset
from torch.utils.data import Dataset, DataLoader
from preprocess_dataset import fix_dataset
import torch
import torch.nn as nn
from CarBehaviorCNN import CarBehaviorCNN

# === CONFIGURAZIONE ===
WINDOW_SIZE = 1
TEST_SPLIT_RATIO = 0.2

# === CARICAMENTO E PULIZIA ===
df = pd.read_csv("../data/vehicle_telemetry_abu36GF2.csv")

df = fix_dataset(df)

# Codifica target
label_encoder = LabelEncoder()
df['result'] = label_encoder.fit_transform(df['result'])  # 0, 1, 2

# Normalizza le feature (tranne la colonna target)
features = df.drop(columns=['result'])
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Ricostruisci DataFrame normalizzato con target
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
df_scaled['result'] = df['result']

# === CREAZIONE DELLE SEQUENZE ===
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        window = data.iloc[i:i+window_size]
        X.append(window.drop(columns=['result']).values)
        y.append(data.iloc[i+window_size]['result'])  # predici in base al record successivo
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, WINDOW_SIZE)

# === SPLIT TRAIN/TEST RISPETTANDO L'ORDINE ===
split_idx = int(len(X) * (1 - TEST_SPLIT_RATIO))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dataset = CarDataset(X_train, y_train)
test_dataset = CarDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CarBehaviorCNN(input_features=X_train.shape[2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        pred = output.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Accuracy: {correct / total:.2%}")

