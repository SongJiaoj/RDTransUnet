# RDTransUnet
Improve the accuracy of image segmentation through residual connection and attention mechanism
# Requirements
Install from the using:`requirements.txt`
```
pip install -r requirements.txt
```
# Training
The first step is to change the settings in the `Config.py` , all the configurations including learning rate, batch size and etc. are in it.
```
python train_model.py
```
# Test
First, change the session name in as the training phase, all the configurations are in `Config.py`.
```
python test_model.py
```
You can get the Dice,IoU,Acc,Sen and Sp scores and the visualization results.
# Environment
Framework: PyTorch 1.11.0.

Language: Python 3.8

CUDA: 11.3

GPU: A40
# Datasets
MoNuSeg、GlaS、DRIVE、CHASEDB1、STARE
