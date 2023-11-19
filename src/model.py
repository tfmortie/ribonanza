"""
Code containing different model architectures and corresponding Lightning modules.
"""
import torch
import lightning.pytorch as pl

from torch import optim, nn
from utils import masked_mae, masked_mse_loss

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
            nn.Linear(n_hidden2, n_out),
            nn.ReLU()
        )
        # ok as long as we don't have high-capacity models
        self.double()
    
    def forward(self, x):
        e = self.embedding(x)
        e = e.view(e.size(0), -1)
        o = self.fc(e)

        return o


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)
    
class ConvLayerNorm(nn.Module):
    def __init__(self, dim, conv1d=True):
        super().__init__()
        if conv1d == True:
            self.norm = nn.Sequential(
                Permute(0, 2, 1), nn.LayerNorm(dim), Permute(0, 2, 1)
            )
        else:
            self.norm = nn.Sequential(
                Permute(0, 2, 3, 1), nn.LayerNorm(dim), Permute(0, 3, 1, 2)
            )

    def forward(self, x):
        return self.norm(x)


class MaskedConvWrapper(nn.Module):
    def __init__(self, ConvModule):
        """
        Works only for Conv1ds that have the same input shape as output shape.
        """
        assert isinstance(ConvModule, nn.Conv1d)
        super().__init__()
        self.conv = ConvModule

    def forward(self, x, mask=None):
        if mask is not None:
            return self.conv(x * mask.unsqueeze(1))
        else:
            return self.conv(x)

# code credits: https://github.com/lucidrains/x-transformers
class GLU(nn.Module):
    def __init__(self, dim, ff_dim, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim, ff_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)

class ConvNext1DBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        dropout=0.1,
        depthwise=True,
        only_one_residual=True,
        only_one_norm=True,
        masked_conv=False,
        glu_ff=False,
        activation="gelu",
    ):
        super().__init__()

        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "swish":
            act = nn.SiLU()

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=(dim if depthwise else 1),
        )
        if masked_conv:
            self.conv = MaskedConvWrapper(self.conv)

        self.norm = ConvLayerNorm(dim)

        project_in = (
            nn.Sequential(nn.Linear(dim, 4 * dim), act)
            if not glu_ff
            else GLU(dim, 4 * dim, act)
        )

        self.pointwise_net = nn.Sequential(
            Permute(0, 2, 1),
            project_in,
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            Permute(0, 2, 1),
        )

        self.one_res = only_one_residual
        self.one_norm = only_one_norm
        if not self.one_norm:
            self.prenorm = ConvLayerNorm(dim)

    def forward(self, x, mask=None):
        z_inter = x
        if not self.one_norm:
            z_inter = self.prenorm(z_inter)

        if isinstance(self.conv, MaskedConvWrapper):
            z_inter = self.conv(z_inter, mask=mask)
        else:
            z_inter = self.conv(z_inter)

        if not self.one_res:
            z_inter = z_inter + x

        z_ff = self.norm(z_inter)
        z_ff = self.pointwise_net(z_ff)
        return z_ff + (x if self.one_res else z_inter)

class Conv1DBaseline(nn.Module):
    def __init__(self, hdim = 32, n_blocks = 10):
        super().__init__()
        self.embedding = nn.Embedding(5, hdim)        

        self.convs = nn.ModuleList([
            ConvNext1DBlock(
                hdim,
                kernel_size=7,
                dropout=0.2,
                depthwise=False,
                only_one_residual=False,
                only_one_norm=False,
                glu_ff=True,
                activation="gelu",
                masked_conv=True,
            ) for _ in range(n_blocks)
        ])

        self.output_head = nn.Linear(hdim, 1)

    
    def forward(self, x):
        mask = x != 0 # for use in conv block B, L

        e = self.embedding(x)
        e = e.permute(0,2,1) # B, L, C -> B, C, L

        for conv in self.convs:
            e = conv(e, mask = mask) # B, C, L -> B, C, L
        
        e = e.permute(0,2,1) # B, C, L -> B, L, C 
        o = self.output_head(e).squeeze(-1) # B, L, C -> B,L,1 -> B, L
        return o

class MTMModel(pl.LightningModule):
    def __init__(self, model, loss = masked_mse_loss, lr = 1e-5):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        x, y, ye, m = batch # x: sequence, y: reactivity, ye: reactivity error, m: mask
        x = x.long()
        y = y.to(self.dtype) # this is my trick to allow for any precision training in lightning

        z = self.model(x) 
        if "weighted" in self.loss.__name__:
            loss = self.loss(z, y, ye, reduction="mean")
        else:
            loss = self.loss(z, y, reduction="mean")
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ye, m = batch # x: sequence, y: reactivity, ye: reactivity error, m: mask
        x = x.long()
        y = y.to(self.dtype)

        z = self.model(x) 
        if "weighted" in self.loss.__name__:
            loss = self.loss(z, y, ye, reduction="mean")
        else:
            loss = self.loss(z, y, reduction="mean")
        mae = masked_mae(z, y, reduction="mean") 
        self.log("val_loss", loss)
        self.log("val_mae", mae)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer