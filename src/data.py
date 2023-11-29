import h5torch
import numpy as np
from lightning import LightningDataModule
import torch
from torch.nn.utils.rnn import pad_sequence
"""
Data utilities to use in training scripts
"""

class RibonanzaDataModule(LightningDataModule):
    def __init__(
        self,
        path, # path to h5torch train file
        batch_size=16, # batch size for model
        n_workers=4, # num workers in dataloader
        split_number = 0, # an integer 0 to 19, indicating which 1/20th (random) fraction of data to use. (for ensembling purposes later)
        in_memory=True, # whether to use h5torch in-memory mode for more-efficient dataloading
    ):
        super().__init__()
        self.n_workers = n_workers 
        self.batch_size = batch_size 
        self.path = path 
        self.in_memory = in_memory 
        self.split_number = split_number


    def setup(self, stage):
        print("Reading data into memory ...")
        f = h5torch.File(self.path).to_dict()
        split_vector = f["0/split"][:]
        train_indices = split_vector != self.split_number
        val_indices = split_vector == self.split_number

        self.train = h5torch.Dataset(f, sample_processor=self.sample_processor, subset=train_indices, in_memory = self.in_memory)

        self.val = h5torch.Dataset(f, sample_processor=self.sample_processor, subset=val_indices, in_memory = self.in_memory)

        self.test = None # has to still be implemented!

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    @staticmethod
    def sample_processor(f, sample):
        c1 = sample["0/bpp_c1"]
        c2 = sample["0/bpp_c2"]
        o = sample["0/bpp_o"]
        shp = len(sample["0/reactivity_2A3"])
        bpp = np.zeros((shp, shp))
        bpp[c1, c2] = o
        bpp[c2, c1] = o
        final_sample = {
            "seq" : sample["0/sequences"],
            "2A3" : sample["0/reactivity_2A3"],
            "DMS" : sample["0/reactivity_DMS"],
            "2A3_e" : sample["0/reactivity_error_2A3"],
            "DMS_e" : sample["0/reactivity_error_DMS"],
            "bpp" : bpp,
            "id" : sample["central"],
            "s_to_n_2A3" : sample["0/s_to_n_2A3"],
            "s_to_n_DMS" : sample["0/s_to_n_DMS"],
            "sn_filter_2A3" : sample["0/sn_filter_2A3"],
            "sn_filter_DMS" : sample["0/sn_filter_DMS"],
        }
        return final_sample


def batch_collater(batch):
    batch_collated = {}
    keys = list(batch[0])
    for k in keys:
        v = [b[k] for b in batch]
        if isinstance(v[0], str):
            batch_collated[k] = v
        elif isinstance(v[0], (int, np.int64)):
            batch_collated[k] = torch.tensor(v)
        elif isinstance(v[0], np.ndarray):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.tensor(np.array(v))
            else:
                batch_collated[k] = pad_sequence(
                    [torch.tensor(t) for t in v], batch_first=True, padding_value=-1
                )
        elif torch.is_tensor(v[0]):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.stack(v)
            else:
                if v[0].dtype == torch.bool:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=False
                    )
                else:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=-1
                    )
    return batch_collated