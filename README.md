# CTRL-F
### **CTRL-F: Pairing Convolution with Transformer for Image Classification via Multi-Level Feature Cross-Attention and Representation Learning Fusion**
This is the official pytorch implementation of the [CTRL-F paper](https://www.arxiv.org/abs/2407.06673).

<hr />

> **Abstract:** *Transformers have captured growing attention in computer vision, thanks to its large capacity and global processing capabilities. However, transformers are data hungry, and their ability
to generalize is constrained compared to Convolutional Neural Networks (ConvNets), especially
when trained with limited data due to the absence of the built-in spatial inductive biases present
in ConvNets. In this paper, we strive to optimally combine the strengths of both convolution
and transformers for image classification tasks. Towards this end, we present a novel lightweight
hybrid network that pairs Convolution with Transformers via Representation Learning Fusion
and Multi-Level Feature Cross-Attention named CTRL-F. Our network comprises a convolution branch and a novel transformer module named multi-level feature cross-attention (MFCA).
The MFCA module operates on multi-level feature representations obtained at different convolution stages. It processes small patch tokens and large patch tokens extracted from these
multi-level feature representations via two separate transformer branches, where both branches
communicate and exchange knowledge through cross-attention mechanism. We fuse the local responses acquired from the convolution path with the global responses acquired from the
MFCA module using novel representation fusion techniques dubbed adaptive knowledge fusion
(AKF) and collaborative knowledge fusion (CKF). Experiments demonstrate that our CTRLF variants achieve state-of-the-art performance, whether trained from scratch on large data
or even with low-data regime. For Instance, CTRL-F achieves top-1 accuracy of 82.24% and
99.91% when trained from scratch on Oxford-102 Flowers and PlantVillage datasets respectively, surpassing state-of-the-art models which showcase the robustness of our model on image
classification tasks. Code at: https://github.com/hosamsherif/CTRL-F.*
<hr />

## Architecture Overview
<div align="center">
<img src="images/CTRL-F.svg" />
</div>

<div align="center">
<img src="images/MFCA.svg" />
</div>

## Installation
Begin by cloning the CTRL-F repository and navigating to the project directory.
```bash
git clone https://github.com/hosamsherif/CTRL-F.git
cd CTRL-F
```

Create a new conda virtual environment.
```bash
conda create -n CTRL_F
conda activate CTRL_F
```

Install [Pytorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) using the following instruction.
```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies.
```bash
pip install -r requirements.txt
```

## Data preparation
Download your training and testing data and structure the data as follows
```bash
/path/to/dataset/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  test/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training

To train CTRL-F models on your dataset, use the below command and specify one of the CTRL-F models

```shell script
python main.py --model CTRLF_B_AKF --batch-size 16 --epochs 200 --data-path /path/to/dataset
```

## Evaluation

To evaluate the model, provide the model checkpoint and the test dataset you want to use for evaluation.

```shell script
python test.py --checkpoint /path/to/model/checkpoint  --model CTRLF_B_AKF --batch-size 32 --data-path /path/to/testset
```

## 📧 Contact
if you have any question, please email `hosamsherif2000@gmail.com` or `hossamsherif@cis.asu.edu.eg`

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
Please consider citing our paper if you find it useful in your research
```
@article{el2024ctrl,
  title={CTRL-F: Pairing Convolution with Transformer for Image Classification via Multi-Level Feature Cross-Attention and Representation Learning Fusion},
  author={EL-Assiouti, Hosam S and El-Saadawy, Hadeer and Al-Berry, Maryam N and Tolba, Mohamed F},
  journal={arXiv preprint arXiv:2407.06673},
  year={2024}
}
```