import torch
from torch import nn
import torch.nn.functional as F

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=2, num_layers=1, d_model=512, nhead=4, dim_feedforward=2048, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x 
    
def custom_loss_1():
    def inner(output, target):
        t = target - output
        t = torch.pow(t, 2).mean()
        return t
    return inner

def custom_loss_2(M=2, alpha=0.1):
    def inner(output, target):
        output_norm = F.normalize(output, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        dist_sq = torch.sum((output_norm - target_norm) ** 2, dim=1)
        
        loss = F.relu(M - dist_sq)
        return loss.mean() * alpha
    return inner