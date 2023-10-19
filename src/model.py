"""
Code containing different models.
"""
import torch.nn as nn

class MTMModel(nn.Module):
    def __init__(self, embedding_dim, n_hidden1, n_hidden2, n_out):
        super().__init__()
        self.embedding = nn.Embedding(5, embedding_dim)        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*n_out, n_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden1),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden2),
            nn.Linear(n_hidden2, n_out)
        )
    
    def forward(self, x):
        e = self.embedding(x)
        e = e.view(e.size(0), -1)
        o = self.fc(e)

        return o