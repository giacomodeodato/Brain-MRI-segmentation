<p align="center">
    <img src="images/examples.png" width="800" title="Examples" />
</p>

# Brain-MRI-segmentation
Tumor segmentation in brain MRI using U-Net \[1\] optimized with the Dice Loss \[2]. The dataset is available from this [repository](https://github.com/giacomodeodato/BrainMRIDataset).

Below are displayed the training curves of the U-Net with 4 blocks of depth, with a fixed number of hidden features equal to 32. The model has been optimized using Adam with a learning rate of 1e-4 for 150 epochs and a learning rate step with gamma = 0.1 at epoch 30. The training code is available in this [notebook](https://github.com/giacomodeodato/Brain-MRI-segmentation/blob/main/BrainMRISegmentation/Brain_MRI_Segmentation.ipynb).

<p align="center">
    <img src="images/training_curves.png" width="400" title="Training Curves" />
</p>

## `UNet.py`

## `DiceLoss.py`
The Dice Loss is a criterion for image segmentation that uses the Sørensen–Dice coefficient between the true segmentation mask and the predicted one \[2\]. It is implemented as follows:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Ctext%7BDice+loss%7D+%26%3D+1+-+%5C+%5Cfrac%7B2%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+p_ig_i%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dp_i+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dg_i%7D%0A%5Cend%7Balign%2A%7D%0A">
</p>

## `MRIDataset.py`
The MRIDataset class is a simple interface for the Dataset class from [brainMRI](https://github.com/giacomodeodato/BrainMRIDataset/tree/main/brainMRI) to be compatible with pytorch DataLoader.

## References
\[1\] **\[Ronneberger, Fischer, & Brox, 2015\]** "U-net: Convolutional networks for biomedical image segmentation". *International Conference on Medical image computing and computer-assisted intervention*.

\[2\] **\[Milletari, Navab, & Ahmadi, 2016\]** "V-net: Fully convolutional neural networks for volumetric medical image segmentation". *International Conference on 3D Vision*.
