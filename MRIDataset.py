import torch
from brainMRI import Dataset

class MRIDataset(torch.utils.data.Dataset):
    
    def __init__(self, train=True, volume=None):
        super(MRIDataset).__init__()
        self.dataset = Dataset(train=train, volume=volume)
    
    def __getitem__(self, index):
        img, mask,_ = self.dataset[index]
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask
    
    def __len__(self):
        return len(self.dataset)