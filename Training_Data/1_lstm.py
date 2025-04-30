import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from preprocess_dataset import fix_dataset

# === 1. Caricamento e preprocessing ===
df = pd.read_csv("../data/vehicle_telemetry_abu36GF2.csv")

df = fix_dataset(df)

label_encoder = LabelEncoder()
df["result"] = label_encoder.fit_transform(df["result"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=["result"]))
y = df["result"].to_numpy()

window_size = 5
X_seq, y_seq = [], []

for i in range(len(X_scaled) - window_size):
    X_seq.append(X_scaled[i:i+window_size])
    y_seq.append(y[i + window_size - 1]) 

X_seq = np.array(X_seq)
y_seq = to_categorical(y_seq, num_classes=4)

train_size = int(0.7 * len(X_seq))
val_size = int(0.20 * len(X_seq))

X_train, y_train = X_seq[:train_size], y_seq[:train_size]
X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

model = Sequential([
    LSTM(64, input_shape=(window_size, X_seq.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(X_train, y_train, epochs=50, batch_size=64,
          validation_data=(X_val, y_val), verbose=1)


# === 5. Valutazione ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")
