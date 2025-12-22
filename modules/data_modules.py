import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

class MethDataset(Dataset):
    def __init__(self, sequence, histone, methylation, coords, apply_log10=False):
        self.sequence = sequence
        self.histone = histone
        self.methylation = methylation
        self.transform = apply_log10
        self.coords = coords
        self.histone_names = ['H3K4me3', 'H3K36me2', 'H3K27me3', 'H3K9me3']

    def __len__(self):
        return self.methylation.shape[0]

    def __getitem__(self, idx):
        
        sequence = torch.from_numpy(self.sequence[idx])
        histone = self.histone.astype(np.float32)

        H3K4me3 = torch.from_numpy(histone[:, :, 0][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 0]+1e-4)[idx])
        H3K36me2 = torch.from_numpy(histone[:, :, 1][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 1]+1e-4)[idx])
        H3K27me3 = torch.from_numpy(histone[:, :, 2][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 2]+1e-4)[idx])
        H3K9me3 = torch.from_numpy(histone[:, :, 3][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 3]+1e-4)[idx])

        methylation = self.methylation[idx]
        coordinates = self.coords[idx]

        return sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates

class MethDataModule(pl.LightningDataModule):
    def __init__(self, npz_path, train_split=0.8, batch_size=32, apply_log10=True):
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.transform = apply_log10
        self.histone_names = ['H3K4me3', 'H3K36me2', 'H3K27me3', 'H3K9me3']
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.data = np.load(self.npz_path, allow_pickle=True)
        split_index = int(self.train_split * self.data['dna'].shape[0]) ### 80% of the data will be for training

        self.train_dataset = MethDataset(sequence = self.data['dna'][:split_index],
                                histone = self.data['histone'][:split_index], 
                                methylation = self.data['methyl'][:split_index],
                                coords = self.data['coords'][:split_index],
                                apply_log10=self.transform)

        self.test_dataset = MethDataset(sequence = self.data['dna'][split_index:],
                                histone = self.data['histone'][split_index:], 
                                methylation = self.data['methyl'][split_index:],
                                coords = self.data['coords'][split_index:],
                                apply_log10=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
