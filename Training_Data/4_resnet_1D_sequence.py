import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from preprocess_dataset import fix_dataset

# === Early Stopping Utility ===
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# === 1. Caricamento e preprocessing ===
df = pd.read_csv("../data/vehicle_telemetry_abu36GF2.csv")
df = fix_dataset(df)

label_encoder = LabelEncoder()
df["result"] = label_encoder.fit_transform(df["result"])

features = df.drop(columns=["result"]).columns
X = df[features].to_numpy()
y = df["result"].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 2. Creazione finestre temporali ===
window_size = 5
X_seq, y_seq = [], []

for i in range(len(X_scaled) - window_size):
    X_seq.append(X_scaled[i:i+window_size])
    y_seq.append(y[i + window_size - 1])

X_seq = np.array(X_seq)  # (samples, seq_len, features)
y_seq = np.array(y_seq)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.long)

# === 3. Dataset e DataLoader ===
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

# === 4. ResNet1D per sequenze temporali ===
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet1DSequence(nn.Module):
    def __init__(self, seq_len, num_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.block1 = ResidualBlock1D(64)
        self.block2 = ResidualBlock1D(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# === 5. Addestramento ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet1DSequence(seq_len=window_size, num_features=X_seq.shape[2], num_classes=len(np.unique(y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
early_stopper = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

# === 6. Valutazione finale ===
from sklearn.metrics import classification_report
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
        all_labels.extend(yb.numpy())

print("\nTest Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
