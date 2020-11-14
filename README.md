# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization

--- 19.12.15 --- <br>
I'm working on editing this script for my own purpose(1 scan type, T1CE).
You can use **model_1channel_mo.py** instead of **model.py** if you have input data with only 1 scan type.

--- all the following statements is from [IAmSuyogJadhav](https://github.com/IAmSuyogJadhav/3d-mri-brain-tumor-segmentation-using-autoencoder-regularization) 

![Keras](https://img.shields.io/badge/Implemented%20in-Keras-red.svg)

![The model architecture](https://www.suyogjadhav.com/images/misc/brats2018_sota_model.png)
<center><b>The Model Architecture</b></center><br /><center>Source: https://arxiv.org/pdf/1810.11654.pdf</center>
<br /><br />

Keras implementation of the paper <b>3D MRI brain tumor segmentation using autoencoder regularization</b> by Myronenko A. (https://arxiv.org/abs/1810.11654). The author (team name: <b>NVDLMED</b>) ranked #1 on the <a href="https://www.med.upenn.edu/sbia/brats2018/" target="_blank">BraTS 2018</a> leaderboard using the model described in the paper.

This repository contains the model complete with the loss function, all implemented end-to-end in Keras. The usage is described in the next section.


