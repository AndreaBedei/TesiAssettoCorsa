import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocess_dataset import fix_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === 1. Caricamento e preprocessamento ===
df = pd.read_csv("../data/vehicle_telemetry_abu36GF2.csv")

df = fix_dataset(df)

# Encode della colonna 'result'
label_encoder = LabelEncoder()
df["result"] = label_encoder.fit_transform(df["result"])
# Mapping: neutro → 0, alto_grip_potenziale_accelerazione → 1, perdita_aderenza → 2

# Split input/output
X = df.drop(columns=["result"])
y = to_categorical(df["result"], num_classes=4)

# Normalizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
train_size = int(0.7 * len(X))
val_size = int(0.20 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# === 2. Modello ===

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Aggiungiamo un layer di dropout per prevenire l'overfitting
    Dense(128, activation='relu'),
    Dropout(0.2),  # Secondo layer di dropout
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # Tre classi
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 3. Addestramento ===
history = model.fit(X_train, y_train, epochs=50,
          validation_data=(X_val, y_val), verbose=1)

# === 4. Valutazione ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")
