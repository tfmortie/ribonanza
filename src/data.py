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
        train_path, # path to h5torch train file
        test_path, # path to h5torch test file
        batch_size=16, # batch size for model
        n_workers=4, # num workers in dataloader
        split_number = 0, # an integer 0 to 19, indicating which 1/20th (random) fraction of data to use. (for ensembling purposes later)
        train_in_memory=True, # whether to use h5torch in-memory mode for more-efficient dataloading
        test_in_memory=True,
    ):
        super().__init__()
        self.n_workers = n_workers 
        self.batch_size = batch_size 
        self.train_path = train_path
        self.test_path = test_path
        self.split_number = split_number
        self.train_in_memory = train_in_memory
        self.test_in_memory = test_in_memory


    def setup(self, stage):
        print("Reading datasets into memory ...")
        f = h5torch.File(self.train_path)
        split_vector = f["0/split"][:]
        train_indices = split_vector != self.split_number
        val_indices = split_vector == self.split_number
        f.close()

        self.train = h5torch.Dataset(self.train_path, sample_processor=self.train_sample_processor, subset=train_indices, in_memory = self.train_in_memory)

        self.val = h5torch.Dataset(self.train_path, sample_processor=self.train_sample_processor, subset=val_indices, in_memory = self.train_in_memory)

        self.test = h5torch.Dataset(self.test_path, sample_processor=self.test_sample_processor, in_memory = self.test_in_memory)

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
    def train_sample_processor(f, sample):
        c1 = sample["0/bpp_c1"]
        c2 = sample["0/bpp_c2"]
        o = sample["0/bpp_o"]
        shp = len(sample["0/reactivity_2A3"])
        bpp = np.zeros((shp, shp))
        bpp[c1-1, c2-1] = o # coordinates are 1-indexed.
        bpp[c2-1, c1-1] = o
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

    @staticmethod
    def test_sample_processor(f, sample):
        c1 = sample["0/bpp_c1"]
        c2 = sample["0/bpp_c2"]
        o = sample["0/bpp_o"]
        shp = len(sample["0/sequences"])
        bpp = np.zeros((shp, shp))
        bpp[c1-1, c2-1] = o # coordinates are 1-indexed.
        bpp[c2-1, c1-1] = o
        final_sample = {
            "seq" : sample["0/sequences"],
            "bpp" : bpp,
            "id" : sample["central"],
            "id_min" : sample["0/id_min"],
            "id_max" : sample["0/id_max"],
            "future" : sample["0/future"],
        }
        return final_sample

class RibonanzaBERTDataModule(LightningDataModule):
    def __init__(
        self,
        path, # path to h5torch train file
        batch_size=128, # batch size for model
        n_workers=4, # num workers in dataloader
        in_memory=True, # whether to use h5torch in-memory mode for more-efficient dataloading
        p = 0.15, # masking probability
    ):
        super().__init__()
        self.n_workers = n_workers 
        self.batch_size = batch_size 
        self.path = path
        self.in_memory = in_memory
        self.p = 0.15

    def setup(self, stage):
        print("Reading datasets into memory ...")
        f = h5torch.File(self.path)
        split_vector = f["0/split"][:]
        train_indices = split_vector == 0
        val_indices = split_vector == 1
        if self.in_memory:
            f = f.to_dict()

        self.train = h5torch.Dataset(f, sample_processor=self.sample_processor, subset=train_indices, in_memory = self.in_memory)

        self.val = h5torch.Dataset(f, sample_processor=self.sample_processor, subset=val_indices, in_memory = self.in_memory)

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

    def sample_processor(self, f, sample):
        c1 = sample["0/bpp_c1"]
        c2 = sample["0/bpp_c2"]
        o = sample["0/bpp_o"]
        shp = len(sample["central"])
        bpp = np.zeros((shp, shp))
        bpp[c1-1, c2-1] = o # coordinates are 1-indexed.
        bpp[c2-1, c1-1] = o

        to_mask = np.random.rand(shp) < self.p
        seq_masked = np.copy(sample["0/sequences"])
        seq_masked[to_mask] = 0

        final_sample = {
            "seq_masked" : seq_masked,
            "seq_orig": sample["0/sequences"],
            "structure" : sample["central"],
            "bpp" : bpp,
        }
        return final_sample


def batch_collater(batch):
    batch_collated = {}
    keys = list(batch[0])
    for k in keys:
        v = [b[k] for b in batch]
        if k == "bpp":
            batch_collated[k] = pad_bpp([torch.tensor(t) for t in v])
        elif isinstance(v[0], str):
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


def pad_bpp(bpps):
    max_ = max([len(b) for b in bpps])
    return torch.stack([torch.nn.functional.pad(
        bpp,
        pad = (0, max_ - len(bpp), 0, max_ - len(bpp)),
        value = -1
    ) for bpp in bpps])