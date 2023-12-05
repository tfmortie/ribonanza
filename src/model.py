"""
Code containing different model architectures and corresponding Lightning modules.
"""
import torch
import lightning.pytorch as pl
from torch import optim, nn
import torch.nn.functional as F
from bio_attention.attention import TransformerEncoder
from bio_attention.embed import DiscreteEmbedding
from einops import rearrange, repeat

class Transformer(pl.LightningModule):
    def __init__(
        self,
        dim = 512,
        depth = 12,
        dropout = 0.2,
        loss = "mae",
        weighted = True,
        target = "both",
        use_bpp = True,
        pos_enc = "rotary",
        lr=1e-4,
        weight_decay=0,
        lr_decay_factor=0.95,
        warmup_steps=500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embed = DiscreteEmbedding(
            5,
            embedding_dim=dim,
            cls=False
        )

        if pos_enc == "rotary":
            plugin_args = {"head_dim" : dim // 8}
        elif pos_enc == "sinusoidal":
            plugin_args = {"dim" : dim}

        self.transformer = TransformerEncoder(
            depth=depth,
            dim=dim,
            nh=8,
            attentiontype="vanilla",
            attention_args={"dropout": dropout},
            plugintype=pos_enc,
            plugin_args=plugin_args,
            only_apply_plugin_at_first=(False if pos_enc == "rotary" else True),
            dropout=dropout,
            glu_ff=True,
            activation="gelu",
        )
        self.use_bpp = use_bpp
        if self.use_bpp:
            self.mask_net = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 8))
            
        self.output_head = nn.Linear(dim, (2 if target == "both" else 1))

        self.loss = Loss(norm = (1 if loss=="mae" else 2), weighted = weighted)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_factor = lr_decay_factor
        self.warmup_steps = warmup_steps
        self.target = target
        self.mae = Loss(norm = 1, weighted = False)

    def forward(self, batch):

        # creating a mask for the transformer to use bpp:
        mask = batch["seq"] != -1
        mask = repeat(mask, "... l -> ... (l2) l", l2=mask.shape[-1])
        if not self.use_bpp:
            mask = (~mask).to(self.dtype).masked_fill(~mask, -float("inf"))
        else:
            bpp = (batch["bpp"][:, :batch["seq"].shape[1], :batch["seq"].shape[1]]).to(self.dtype)
            bpp = self.mask_net(bpp.unsqueeze(-1))
            mask = bpp.masked_fill(~mask.unsqueeze(-1), -float("inf"))
            mask = rearrange(mask, "b l1 l2 h -> b h l1 l2")

        x = self.embed(batch["seq"])
        y = self.transformer(x, mask = mask, pos = None)
        return self.output_head(y)

    def training_step(self, batch, batch_idx):
        y = self(batch)
        
        if self.target == "both":
            targets = torch.stack([batch["2A3"], batch["DMS"]], dim = -1)[:, :y.shape[1]].to(self.dtype)
            weights = torch.stack([batch["2A3_e"], batch["DMS_e"]], dim = -1)[:, :y.shape[1]].to(self.dtype)
        else:
            targets = batch[self.target].unsqueeze(-1)[:, :y.shape[1]].to(self.dtype)
            weights = batch[self.target+"_e"].unsqueeze(-1)[:, :y.shape[1]].to(self.dtype)
        
        loss = self.loss(y, torch.clip(targets, 0, 1), torch.clip(1 / weights, 0, 10))

        self.log("train_loss", loss, sync_dist = True)

        return loss


    def validation_step(self, batch, batch_idx):
        y = self(batch)
        
        if self.target == "both":
            targets = torch.stack([batch["2A3"], batch["DMS"]], dim = -1)[:, :y.shape[1]].to(self.dtype)
            weights = torch.stack([batch["2A3_e"], batch["DMS_e"]], dim = -1)[:, :y.shape[1]].to(self.dtype)
        else:
            targets = batch[self.target].unsqueeze(-1)[:, :y.shape[1]].to(self.dtype)
            weights = batch[self.target+"_e"].unsqueeze(-1)[:, :y.shape[1]].to(self.dtype)
        
        loss = self.loss(y, torch.clip(targets, 0, 1), torch.clip(1 / weights, 0, 10))
        mae = self.mae(y, torch.clip(targets, 0, 1), torch.clip(1 / weights, 0, 10))
        self.log("val_loss", loss, sync_dist = True)
        self.log("val_mae", mae, sync_dist = True)

    def predict_step(self, batch, batch_idx):
        y = self(batch)
        y = torch.clip(y, 0, 1)

        ids = []
        preds_2A3 = []
        preds_DMS = []
        for ix in range(len(y)):
            id_min = batch["id_min"][ix]
            id_max = batch["id_max"][ix]
            id_range_sample = torch.arange(id_min, id_max+1)
            y_sample = y[ix, :(id_max+1-id_min)]

            ids.append(id_range_sample)
            if self.target == "both":
                preds_2A3.append(y_sample[:, 0])
                preds_DMS.append(y_sample[:, 1])
            elif self.target == "DMS":
                preds_DMS.append(y_sample[:, 0])
            elif self.target == "2A3":
                preds_2A3.append(y_sample[:, 0])
        
        return [
            torch.cat(ids),
            (torch.cat(preds_DMS).astype(torch.float32) if len(preds_DMS) > 0 else torch.tensor([])),
            (torch.cat(preds_2A3).astype(torch.float32) if len(preds_2A3) > 0 else torch.tensor([])),
        ]


    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        lambd = lambda epoch: self.lr_decay_factor
        lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # warm up lr
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

class Loss(nn.Module):
    def __init__(self, norm=1, weighted = True):
        super().__init__()
        self.norm = norm
        self.weighted = weighted

    def forward(self, input, target, weight):
        mask = torch.isnan(target)
        out = torch.abs(input[~mask]-target[~mask])**self.norm
        if self.weighted:
            out *= weight[~mask]
        return out.mean()


class RibonanzaBERT(pl.LightningModule):
    def __init__(
        self,
        dim = 512,
        depth = 12,
        lr = 1e-4,
        dropout = 0.2,
        use_bpp = True,
        pos_enc = "rotary",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embed = DiscreteEmbedding(
            5,
            embedding_dim=dim,
            cls=False
        )

        if pos_enc == "rotary":
            plugin_args = {"head_dim" : dim // 8}
        elif pos_enc == "sinusoidal":
            plugin_args = {"dim" : dim}

        self.transformer = TransformerEncoder(
            depth=depth,
            dim=dim,
            nh=8,
            attentiontype="vanilla",
            attention_args={"dropout": dropout},
            plugintype=pos_enc,
            plugin_args=plugin_args,
            only_apply_plugin_at_first=(False if pos_enc == "rotary" else True),
            dropout=dropout,
            glu_ff=True,
            activation="gelu",
        )
        self.use_bpp = use_bpp
        if self.use_bpp:
            self.mask_net = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 8))
            
        self.mask_output_head = nn.Linear(dim, 4)
        self.struct_output_head = nn.Linear(dim, 3)

        self.lr = lr

    def forward(self, batch):
        # creating a mask for the transformer to use bpp:
        mask = batch["seq_masked"] != -1
        mask = repeat(mask, "... l -> ... (l2) l", l2=mask.shape[-1])
        if not self.use_bpp:
            mask = (~mask).to(self.dtype).masked_fill(~mask, -float("inf"))
        else:
            bpp = (batch["bpp"][:, :batch["seq_masked"].shape[1], :batch["seq_masked"].shape[1]]).to(self.dtype)
            bpp = self.mask_net(bpp.unsqueeze(-1))
            mask = bpp.masked_fill(~mask.unsqueeze(-1), -float("inf"))
            mask = rearrange(mask, "b l1 l2 h -> b h l1 l2")

        x = self.embed(batch["seq_masked"])
        y = self.transformer(x, mask = mask, pos = None)
        return self.mask_output_head(y), self.struct_output_head(y)

    def training_step(self, batch, batch_idx):
        mlm_preds, struct_preds = self(batch)
        
        mlm_loss = F.cross_entropy(mlm_preds[batch["seq_masked"] == 0], batch["seq_orig"][batch["seq_masked"] == 0]-1)
        
        struct_indices = batch["structure"] != -1
        if struct_indices.sum() > 1:
            struct_loss = F.cross_entropy(struct_preds[struct_indices], batch["structure"][struct_indices])
        else:
            struct_loss = struct_preds.sum() * 0 # I would put 0 here but torch multi-gpu mode does not like this.
        
        loss = mlm_loss + struct_loss
        
        self.log("train_loss", loss, sync_dist = True, batch_size = len(batch["seq_masked"]))
        return loss


    def validation_step(self, batch, batch_idx):
        mlm_preds, struct_preds = self(batch)
        
        mlm_loss = F.cross_entropy(mlm_preds[batch["seq_masked"] == 0], batch["seq_orig"][batch["seq_masked"] == 0]-1)
        
        struct_indices = batch["structure"] != -1
        if struct_indices.sum() > 1: # take into account that not every sample has these labels, so a batch may exist that does not have this loss
            struct_loss = F.cross_entropy(struct_preds[struct_indices], batch["structure"][struct_indices])
        else:
            struct_loss = struct_preds.sum() * 0
        
        loss = mlm_loss + struct_loss

        self.log(
            "val_mlm_loss",
            mlm_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["seq_masked"]),
            sync_dist=True,
            )

        self.log(
            "val_struct_loss",
            struct_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["seq_masked"]),
            sync_dist=True,
            )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["seq_masked"]),
            sync_dist=True,
            )


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer