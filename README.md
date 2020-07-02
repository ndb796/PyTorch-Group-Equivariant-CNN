### PyTorch Group Equivariant CNN

* This repository provides PyTorch implementations for [Group Equivariant CNN (G-CNN)](https://arxiv.org/abs/1602.07576).
* This repository shows fine accuracies which are higher than the accuracies in the original papers.
* If you have questions, please send an e-mail to me (dongbinna@postech.ac.kr) or make an issue.

### Experiment Settings

* The basic source codes used in this repository follows the source codes used in [PyTorch GrouPy Examples](https://github.com/adambielski/pytorch-gconv-experiments).
    * However, my repository contains the essential library, thus there is no need to install additional libraries.
* Architectures: ResNet-18, ResNet-50
* Dataset: CIFAR-10 (10 classes), AIGS-10 (10 classes)
* Training batch size: 128
* Weight decay: 0.0005
* Momentum: 0.9
* Data augmentation: Random crop, Random horizontal flip
* Input normalization
* Learning rate adjustment
  1) 0.1 for epoch [0, 50)
  2) 0.01 for epoch [50, 100)
  3) 0.001 for epoch [100, 150)
  4) 0.0001 for epoch [150, 200)
  5) 0.00001 for epoch [200, 250)

### Training

#### 1. Group P4M (Original Paper)

* The train command example (Single GPU)
<pre>
python3 train.py --n_epochs 250 --checkpoint ResNet18_P4M --lr=0.1
</pre>
* The train results

||CIFAR-10|AIGS-10|
|------|---|---|
|ResNet18|94.44%|47.81%|
|ResNet50|xx.xx%|xx.xx%|

#### 2. Group P4M with Mixup (My Contribution)

* The Mixup method was proposed by Hongyi Zhang in [ICLR 2018](https://arxiv.org/abs/1710.09412).
* The train command example (Single GPU)
<pre>
python3 train.py --n_epochs 250 --checkpoint ResNet18_P4M_Mixup --lr=0.1 --mixup
</pre>

* The train results

||CIFAR-10|AIGS-10|
|------|---|---|
|ResNet18|95.89%|xx.xx%|
|ResNet50|xx.xx%|xx.xx%|

### Testing

* All pre-trained models are provided in this repository :)
<pre>
python3 test.py --n_epochs 180 --checkpoint ResNet18_P4M_Mixup
</pre>
