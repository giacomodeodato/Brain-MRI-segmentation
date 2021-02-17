import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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

    def __init__(self, in_channels=3, out_channels=1, n_features=32, depth=5):
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
    
    def push_activation(self):
        def hook(module, input, output):
            self.activations.append(output)
        return hook
    
    def pop_activation(self):
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