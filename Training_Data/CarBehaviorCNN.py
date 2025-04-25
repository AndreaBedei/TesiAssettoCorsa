import torch.nn as nn
import torch.nn.functional as F

class CarBehaviorCNN(nn.Module):
    def __init__(self, input_features, num_classes=3):
        super(CarBehaviorCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, window_size, features) -> permute to (batch, features, window_size)
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.pool(x).squeeze(-1)  # shape: (batch, channels)
        x = self.dropout(x)
        x = self.fc(x)
        return x
