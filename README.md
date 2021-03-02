<p align="center">
    <img src="images/examples.png" width="800" title="Examples" />
</p>

# Brain-MRI-segmentation
Tumor segmentation in brain MRI using U-Net \[1\] optimized with the Dice Loss \[2]. The dataset is available from this [repository](https://github.com/giacomodeodato/BrainMRIDataset).

Below are displayed the training curves of the U-Net with 4 blocks of depth, with a fixed number of hidden features equal to 32. The model has been optimized using Adam with a learning rate of 1e-4 for 150 epochs and a learning rate step with gamma = 0.1 at epoch 30.

<p align="center">
    <img src="images/training_curves.png" width="400" title="Training Curves" />
</p>

## `UNet.py`

## `DiceLoss.py`

## `MRIDataset.py`

## References
\[1\] **\[Ronneberger, Fischer, & Brox, 2015\]** "U-net: Convolutional networks for biomedical image segmentation". *International Conference on Medical image computing and computer-assisted intervention*.

\[2\] **\[Milletari, Navab, & Ahmadi, 2016\]** "V-net: Fully convolutional neural networks for volumetric medical image segmentation". *International Conference on 3D Vision*.
