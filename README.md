# Enhancing Unsupervised Person Re-Identification via Adaptive Clustering and Sample Expansion

## Getting Started
### Installation
```shell
git clone https://github.com/Zujhie/ACSE.git
cd ACSE
python setup.py develop
```

### Preparing Datasets
```shell
cd examples && mkdir data
```

The directory should look like
```
acse/examples/data
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── dukemtmcreid
│   └── DukeMTMC-reID
```

### Training

For Market-1501:
```
CUDA_VISIBLE_DEVICE=0,1,2,3 \
python examples/acse_train_usl.py \
-b 256 -a resnet50 -d market1501 --iters 200 \
--momentum 0.1 --eps 0.4 --nums-instances 16 --epochs 70
```

For DukeMTMC-reID:
```
CUDA_VISIBLE_DEVICE=0,1,2,3 \
python examples/acse_train_usl.py \
-b 256 -a resnet50 -d dukemtmcreid --iters 200 \
--momentum 0.1 --eps 0.4 --nums-instances 16 --epochs 50
```

For MSMT17:
```
CUDA_VISIBLE_DEVICE=0,1,2,3 \
-b 256 -a resnet50 -d msmt17 --iters 400 \
--momentum 0.1 --eps 0.4 --nums-instances 16 --epochs 50
```


## Testing 

For Market-1501:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test.py \
-d market1501 --resume $PATH
```
For DukeMTMC-reID:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test.py \
-d dukemtmcreid --resume $PATH
```
For MSMT17:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test.py \
-d msmt17 --resume $PATH
```


## Acknowledgement
Some parts of the code is borrowed from [Cluster Contrast](https://github.com/alibaba/cluster-contrast-reid) 

## Citation
If you find this code useful for your research, please consider citing our paper
