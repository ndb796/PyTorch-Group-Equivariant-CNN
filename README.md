### PyTorch Group Equivariant CNN

* This repository provides PyTorch implementations for [Group Equivariant CNN (G-CNN)](https://arxiv.org/abs/1602.07576).
* This repository shows fine accuracies which are higher than the accuracies in the original papers.
* If you have questions, please send an e-mail to me (dongbinna@postech.ac.kr) or make an issue.
* This repository is a final project result in the AIGS538 class in POSTECH, South Korea.

### Experiment Settings

* The basic source codes used in this repository follows the source codes used in [PyTorch GrouPy Examples](https://github.com/adambielski/pytorch-gconv-experiments).
    * However, my repository contains the essential library, thus there is no need to install additional libraries.
* Architectures: ResNet18, ResNet50
* Dataset: CIFAR (10 classes), AIGS (10 classes)
* Training batch size: 128
* Weight decay: 0.0005
* Momentum: 0.9
* Data augmentation: Random crop, Random horizontal flip
* Input normalization
* Learning rate adjustment
  1) 0.01 for epoch [0, 40)
  2) 0.001 for epoch [40, 90)
  3) 0.0001 for epoch [90, 140)
  4) 0.00001 for epoch [140, 180)
  5) 0.000001 for epoch [180, 220)
  6) 0.0000001 for epoch [220, 250)

### Training

#### 1. Group P4M (Original Paper)

* The train command examples (with a single GPU)
<pre>
# ResNet18 on CIFAR
python3 train.py --model ResNet18 --dataset CIFAR --checkpoint ResNet18_P4M_on_CIFAR
# ResNet18 on AIGS
python3 train.py --model ResNet18 --dataset AIGS --checkpoint ResNet18_P4M_on_AIGS
# ResNet50 on CIFAR
python3 train.py --model ResNet50 --dataset CIFAR --checkpoint ResNet50_P4M_on_CIFAR
# ResNet50 on AIGS
python3 train.py --model ResNet50 --dataset AIGS --checkpoint ResNet50_P4M_on_AIGS
</pre>
||CIFAR-10|AIGS-10|
|------|---|---|
|ResNet18|94.44%|47.81%|
|ResNet50|xx.xx%|xx.xx%|
* [Trained model download]

#### 2. Group P4M with Mixup (My Contribution)

* The Mixup method was proposed by Hongyi Zhang in [ICLR 2018](https://arxiv.org/abs/1710.09412).
* The train command examples (with a single GPU)
<pre>
# ResNet18 on CIFAR
python3 train.py --model ResNet18 --dataset CIFAR --checkpoint ResNet18_P4M_Mixup_on_CIFAR --mixup
# ResNet18 on AIGS
python3 train.py --model ResNet18 --dataset AIGS --checkpoint ResNet18_P4M_Mixup_on_AIGS --mixup
# ResNet50 on CIFAR
python3 train.py --model ResNet50 --dataset CIFAR --checkpoint ResNet50_P4M_Mixup_on_CIFAR --mixup
# ResNet50 on AIGS
python3 train.py --model ResNet50 --dataset AIGS --checkpoint ResNet50_P4M_Mixup_on_AIGS --mixup
</pre>
||CIFAR-10|AIGS-10|
|------|---|---|
|ResNet18|95.89%|xx.xx%|
|ResNet50|xx.xx%|xx.xx%|
* [Trained model download]

### Testing

* All pre-trained models are provided in this repository :)
<pre>
python3 test.py --dataset CIFAR --checkpoint ResNet18_P4M_Mixup_on_CIFAR
</pre>
