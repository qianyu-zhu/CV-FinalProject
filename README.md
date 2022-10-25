# FrameWork
## Here is a brief introduction for readers.

This respository is the final project for CSCI-GA.2271, computer vision. The course web is [link](https://cs.nyu.edu/~fergus/teaching/vision/index.html).

## Data
  - For this project, we use the Semantic Drone Dataset from the Institute of Computer Graphics and Vision;
  - It consists of 400 images from nadir (bird’s eye) view acquired at an altitude of 5 to 30 meters above the ground;
  - To carry on semantic segmentation tasks, we use pixel-accurate annotation of 24 classes (1 unlabeled class) to train and test models.

## Model
  - It will contain all the models we need;
  - We use simpleUnet.py and unet_advanced.py, which are two advanced version of venilla Unet;
  - We use enet.py, which is the pyfile for ENet, more accurate than Unet.

## Prune
  - Here are all the pruning techniques we tested, the comparison can be found in the [paper](https://github.com/qianyu-zhu/CV-Final_Project/blob/1050177ca44789f43b1fb7055d0a9e08f0d7fb6a/Efficiency%20in%20Semantic%20Segmentation.pdf).

## Other
  - Some important help functions included in utils.py and DataUtils.py. The full presentation slides can be find [here](https://github.com/qianyu-zhu/CV-Final_Project/blob/main/CV-final%20presentation.pdf).





