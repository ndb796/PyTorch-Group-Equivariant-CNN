### PyTorch Group Equivariant CNN

* This repository provides PyTorch implementations for [Group Equivariant CNN (G-CNN)](https://arxiv.org/abs/1602.07576).
* This repository shows fine accuracies which are higher than the accuracies in the original papers.
* If you have questions, please send an e-mail to me (dongbinna@postech.ac.kr) or make an issue.

### Experiment Settings

* The basic source codes used in this repository follows the source codes used in [PyTorch GrouPy Examples](https://github.com/adambielski/pytorch-gconv-experiments).
    * However, my repository contains the essential library, thus there is no need to install additional library.
* Architectures: ResNet-18, ResNet-50
* Dataset: CIFAR-10 (10 classes), AIGS-10 (10 classes)
* Training batch size: 128
* Weight decay: 0.0005
* Momentum: 0.9
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

||ResNet18|ResNet50|
|------|---|---|
|Test accuarcy|94.44%|xx.xx%|

#### 2. Group P4M with Mixup (My Contribution)

* The train results

||ResNet18|ResNet50|
|------|---|---|
|Test accuarcy|95.89%|xx.xx%|
