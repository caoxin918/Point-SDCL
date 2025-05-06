# Point-SDCL
# Note: If your work uses this algorithm or makes improvements based on it, please be sure to cite this paper. Thank you for your cooperation.

# 注意：如果您的工作用到了本算法，或者基于本算法进行了改进，请您务必引用本论文，谢谢配合



# Point-SDCL: Self-Distillation Contrastive Learning via Positive-Negative Interaction and Probabilistic Modeling

Xin Cao, Deyu Ma, Huan Xia, Jia Zhang, Xingxing Hao, Linzhi Su and Kang Li※


## Requirements:

Make sure the following environments are installed.

```
python=3.8.13
pytorch==1.12.1
torchvision=0.13.1
cudatoolkit=11.3.1
matplotlib=3.5.2
```

## Datasets 

We use ModelNet40 provided by [Princeton](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) for pretrain.

Run the command below to download all the datasets (ModelNet40, ScanObjectNN, ShapeNetPart) to reproduce the results.

```bash
cd datasets
source download_data.sh
```


## Training

```bash
pyton train_pre.py
```

## Citation

```

```

## Acknowledgements

We would like to thank and acknowledge referenced codes from the following repositories:

https://github.com/WangYueFt/dgcnn

https://github.com/charlesq34/pointnet

https://github.com/AnTao97/dgcnn.pytorch
