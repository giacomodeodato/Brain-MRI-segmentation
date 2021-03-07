import torch
import torch.nn as nn
from typing import Callable

class UNetBlock(nn.Module):
    """Block of layers used to build U-Net.
    
    The block is made of the following layers:
        nn.Conv2d
        nn.BatchNorm2d
        nn.ReLU
        nn.Conv2d
        nn.BatchNorm2d
        nn.ReLU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes the block of layers.
        
        The number of internal features is fixed to be equal to the
        number of output channels.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        """

        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UNet(nn.Module):
    """U-Net, neural network architecture for image segmentation [1].

    References
    ----------
        [1] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: 
        Convolutional networks for biomedical image segmentation. In International
        Conference on Medical image computing and computer-assisted intervention 
        (pp. 234-241). Springer, Cham
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, 
                    n_features: int = 32, depth: int = 5) -> None:
        """Initializes the neural network.
        
        The architecture is defined iteratively using a stack of layers' activations
        and forward hooks.
        The depth parameter can be used to specify the number of layers of the encoder
        section of U-Net.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels.
            Default is 3.
        out_channels : int
            Number of output channels.
            Default is 1.
        n_features : int, optional
            Number of hidden features of each layer.
            Default is 32.
        depth : int, optional
            Depth of the architecture.
            Default is 5.
            
        """

        super(UNet, self).__init__()

        self.layers = []
        self.activations = []
        
        self.layers.append(UNetBlock(in_channels, n_features))
        self.layers[-1].register_forward_hook(self.push_activation())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        for _ in range(depth-2):
            self.layers.append(UNetBlock(n_features, n_features))
            self.layers[-1].register_forward_hook(self.push_activation())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layers.append(UNetBlock(n_features, n_features))
        
        for _ in range(depth-1):
            self.layers.append(nn.ConvTranspose2d(
                n_features, n_features, kernel_size=2, stride=2, bias=False
            ))
            self.layers[-1].register_forward_hook(self.pop_activation())
            self.layers.append(UNetBlock(n_features*2, n_features))
            
        self.layers.append(nn.Conv2d(
            in_channels=n_features, out_channels=out_channels, kernel_size=1
        ))
        
        self.layers = nn.Sequential(*self.layers)
    
    def push_activation(self) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
        """Returns a hook that pushes the layer activation in the activation stack."""

        def hook(module, input, output):
            self.activations.append(output)
        return hook
    
    def pop_activation(self) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns a hook that pops an activation from activation stack
        and concatenates it to the layer output."""

        def hook(module, input, output):
            return torch.cat(
                (output, self.activations.pop()),
                dim=1
            )
        return hook

    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x