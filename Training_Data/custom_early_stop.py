from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import precision_score

class CustomEarlyStopping(Callback):
    def __init__(self, validation_data, patience=5):
        super(CustomEarlyStopping, self).__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.best_score = -np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        precision_per_class = precision_score(y_val_true, y_val_pred, average=None, zero_division=0)
        mean_precision = np.mean(precision_per_class)

        print(f"Epoch {epoch + 1}: Mean Precision = {mean_precision:.4f}")

        if mean_precision > self.best_score:
            self.best_score = mean_precision
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Early stopping triggered!")
                self.model.stop_training = True