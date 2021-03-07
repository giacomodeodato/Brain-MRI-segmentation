import torch
from brainMRI import Dataset

class MRIDataset(torch.utils.data.Dataset):
    """Creates a MRI dataset compatible with pytorch DataLoader."""
    
    def __init__(self, train: bool = True, volume: str = None) -> None:
        """Initializes the MRI Dataset.
        
        This dataset is based on the MRI dataset implemented in [1] and 
        the data available from kaggle [2].

        Parameters
        ----------
        train : bool, optional
            Specifies which split of the data to use.
            Default is True, uses 80% of the samples.
        volume : str, optional
            The MRI volume to select among "pre-contrast", "FLAIR" and 
            "post-contrast". Default is None, uses all the volumes.

        References
        ----------
            [1] https://github.com/giacomodeodato/BrainMRIDataset

            [2] https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
        """

        super(MRIDataset).__init__()
        self.dataset = Dataset(train=train, volume=volume)
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        """Returns the sample at the specified index.

        A sample is a tuple (image, mask) made of the volume(s) of the
        sample and the corresponding segmentation mask.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Dataset sample made of volume and segmentation mask.
        """

        img, mask,_ = self.dataset[index]
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask
    
    def __len__(self) -> int:
        """Returns the length of the dataset."""

        return len(self.dataset)