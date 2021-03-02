import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Creates a criterion for image segmentation using the Sørensen–Dice 
    coefficient between the true segmentation mask and the predicted one [1].
    
    References
    ----------
        [1] Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). V-net: Fully 
        convolutional neural networks for volumetric medical image segmentation.
        In 2016 fourth international conference on 3D vision (3DV) (pp. 565-571).
        IEEE.
    """

    def __init__(self) -> None:
        """Initializes the loss function.

        The computation of the Sørensen–Dice coefficient is performed using
        a smoothing factor equal to one for numerical stability.

        Returns
        -------
        DiceLoss()
            instance of DiceLoss
        """

        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred : torch.Tensor, y_true : torch.Tensor) -> torch.Tensor:
        """Computes the Dice loss between two tensors.

        The two tensors must have the same size.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted segmentation mask.
        y_true : torch.Tensor
            The true segmentation mask.

        Returns
        -------
        torch.Tensor
            Dice loss between y_pred and y_true.

        Raises
        ------
        AssertionError
            If the input tensors have different sizes.
        """

        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc