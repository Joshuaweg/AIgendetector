import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Classifier(nn.Module):
    def __init__(self,input_dim,hidden_dim, num_layers, num_classes):
        super(Classifier, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=hidden_dim)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        pooled_x = x.mean(dim=0)
        output= self.fc(pooled_x)
        return output.unsqueeze(0)