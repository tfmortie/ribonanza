"""
Code containing different model architectures and corresponding Lightning modules.
"""
import torch
import lightning.pytorch as pl

from torch import optim, nn
from utils import mse_loss

class MTM(nn.Module):
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
        # ok as long as we don't have high-capacity models
        self.double()
    
    def forward(self, x):
        e = self.embedding(x)
        e = e.view(e.size(0), -1)
        o = self.fc(e)

        return o

class MTMModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y, m = batch
        z = self.model(x) 
        loss = mse_loss(z, y, reduction="mean")
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)

        return optimizer