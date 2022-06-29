### Change My Pet ![visitors](https://visitor-badge.glitch.me/badge?page_id=mohd-muzamil.IrisDashboard)

Experimental project to replace image of a dog/cat with a GAN(Generative adversarial network) generated image with the help of segmentation masks.
 
[Publication](https://link.springer.com/chapter/10.1007/978-3-030-88942-5_21)

# Results:
<div align="center">
    <a>
        <img src="https://github.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/blob/master/assets/example2.png?raw=true" />
    </a>
</div>
<hr />
<div align="center">
    <a>
        <img src="https://github.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/blob/master/assets/example1.png?raw=true">
    </a>
</div>
<br />


# Exciting Gifs:
<div align="center">
    <a>
        <img src="https://raw.githubusercontent.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/master/assets/gif2.gif" width="500" height="500"/>
    </a>
</div>
<br />
<div align="center">
    <a>
        <img src="https://raw.githubusercontent.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/master/assets/gif1.gif" width="500" height="500"/>
    </a>
</div>

 

<br />

 
The results are quite interesting and the model is learning to generate images in the required segment. Regularization and ensembling losses have given more accurate results but we are still exploring other techniques. Our next step is to include Discriminator so that the images generated look more real. Generate a pull request if you want to conrtribute to the project.


### Steps
- Clone this repo.
- To run the code, please download the pretrained pytorch weights first. [Pretrained Weights](https://github.com/ivclab/BigGAN-Generator-Pretrained-Pytorch/releases/tag/v0.0.0)
```shell
    biggan512-release.pt    # download this for generating 512*512 images
```
- Upload the biggan512-release.pt file to your google drive.
- Open MSE_average_losses_clean.ipynb file in Google Colab or your Jupyter Notebook and run it. Comments are added to the file as needed.

 

# References paper: 
https://arxiv.org/abs/1809.11096 <br>
https://github.com/ivclab/BIGGAN-Generator-Pretrained-Pytorch <br>
https://pytorch.org/hub/pytorch_vision_fcn_resnet101/ <br>
https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/ <br>








