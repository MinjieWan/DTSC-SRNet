# DSTC-SRNet
We provide the dataset of our work. You can download it by the following link: [dataset](https://pan.baidu.com/s/12EC2qY9ZBTGvlB386uLFtA) code:mkdn

# Installation

This repository is built in PyTorch 2.0.0 (Python3.10, CUDA12.6).
Follow these intructions

1. Make conda environment
```
conda create -n pytorch2.0 python=3.10
conda activate pytorch2.0
```

2. Install dependencies
```
conda install pytorch=2.0 torchvision -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python tqdm
pip install numpy scipy lpips pillow
```

# Usage

Download and release DTSC.zip. 
Setup the required parameters.
Run main.py for training or testing.

