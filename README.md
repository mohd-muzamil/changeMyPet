# ChangeMyPet
It combines two separately trained models(Biggan and image segmentation) into a single pipeline.
The models are described below.
# DeepLabCut v3
Deeplabv3-ResNet101 is contructed by a Deeplabv3 model with a ResNet-101 backbone. The pre-trained model has been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
Their accuracies of the pre-trained models evaluated on COCO val2017 dataset are listed below.

| Model structure | Mean IOU | Pixelwise Accuracy |
| ------ | ------ | ------ |
| deeplabv3_resnet101 | 67.4 | 92.4 |


`import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()`

All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N, 3, H, W), where N is the number of images, H and W are expected to be at least 224 pixels. The images have to be loaded in to a range of [0, 1] and then normalized using **mean = [0.485, 0.456, 0.406]** and **std = [0.229, 0.224, 0.225].**
The model returns an OrderedDict with two Tensors that are of the same height and width as the input Tensor, but with 21 classes. output['out'] contains the semantic masks, and output['aux'] contains the auxillary loss values per-pixel. In inference mode, output['aux'] is not useful. So, output['out'] is of shape **(N, 21, H, W)**. More documentation can be found [here](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).
# BigGAN Generators with Pretrained Weights in Pytorch 
Pytorch implementation of the generator of Large Scale GAN Training for High Fidelity Natural Image Synthesis (BigGAN). 

# Download Pretrained Weights 
The Pretrained weights can be downloaded from the latest release. [link](
https://github.com/ivclab/BigGAN-Generator-Pretrained-Pytorch/releases/latest) 

# Dependencies 
dependencies:
  - python=3.6
  - cudatoolkit=10.0
  - pytorch
  - torchvision
  - scipy

Please also refer to the environment.yml file. 

# Demo 
- To run the code, please download the pretrained weights first.
```shell 
python demo.py -w <PRETRAINED_WEIGHT_PATH> [-s IMAGE_SIZE] [-c CLASS_LABEL] [-t TRUNCATION] 
python demo.py -w ./biggan512-release.pt -s 512 -t 0.3 -c 156 
python demo.py -w ./biggan256-release.pt -s 256 -t 0.02 -c 11 
python demo.py --pretrained_weight ./biggan128-release.pt --size 128 --truncation 0.2 --class_label 821 
``` 
- Valid image size: 128, 256, 512
- Valid class label: 0~999
- Valid truncation: 0.02~1.0


# Results 
|![alt text](./assets/p1.png)|
|:--:|
|*class 156 (512 x 512)*|
|![alt text](./assets/p2.png)|
|*class 11 (512 x 512)*|
|![alt text](./assets/p3.png)|
|*class 821 (512 x 512)*|


# Pretrained Weights 
The pretrained weights are converted from the tensorflow hub modules: 
- https://tfhub.dev/deepmind/biggan-128/2  
- https://tfhub.dev/deepmind/biggan-256/2 
- https://tfhub.dev/deepmind/biggan-512/2  


# References 
paper: https://arxiv.org/abs/1809.11096

https://github.com/ajbrock/BigGAN-PyTorch

# Contact 

Please feel free to leave suggestions or comments to [Tsung-Hung Hsieh](https://github.com/nemothh)(andrewhsiehth@gmail.com).
